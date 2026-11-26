#!/usr/bin/env python3
"""Checks a fusion region's accepted overlay on disk before the e2e workflow banks it.

    python3 fusion_overlay_check.py --overlay <accepted overlay dir> --initial <the overlay it was built on> \
        --root <EVAL_DIR>

Prints {"ok", "problems", "manifest_sha256", "files"} as one JSON line and exits 0 (ok) or 1. The workflow
script cannot read files, so when an integrate A/B accepts a fusion region (KernelFusion switched on) it
runs this through one agent and banks the region only on "ok": true.

1. it is a directory inside the run dir with a sitecustomize.py;
2. every .py in it compiles (compiled only, never imported or run here);
3. every absolute path it names under the run's parent dir (the fused kernels it loads) exists;
4. existing optimizations are kept: every file of the initial overlay is present and byte-identical,
   and the initial sitecustomize.py is either kept as is or copied under another name that the new
   sitecustomize.py imports. The merged injection manifest (_overlay_manifest.json) is the one exception:
   stacking a region must add its own entries, so it is checked by key instead (every initial
   module / rebind / marker / capture / installer entry is kept unchanged and only once, and no new
   module, rebind or installer target overlaps an initial one -- the conflict rule of
   overlay_setup.py merge), and every other initial field is unchanged. One overlap is allowed: a new
   rebind of <module>:<attr> on a module the initial overlay replaces, when <attr>'s definition in that
   patched file is AST-identical to the installed module's (the rebind then replaces nothing the initial
   overlay changed): every binding of it in its scope, including nested blocks, imports and module-level
   writes such as C.m = ... or setattr(C, "m", ...), is compared. The installed source is located without
   importing anything (the top-level package's spec, which an editable install's finder resolves, then
   sys.path); when it cannot be found or parsed, or <attr> is missing on either side, the overlap is still a
   conflict;
5. it is the overlay the A/B measured: an integrate_result.json accepted exactly this directory with a
   complete A/B, and no code in it (_CODE_SUFFIXES) changed after that A/B's candidate leg finished. The
   A/B's own records (e2e_validation.json, manifests.json, ...) land here after the A/B and are not
   overlay content.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# Benchmark / correctness output the integrator writes next to the overlay sources; not part of the overlay.
_OVERLAY_OUTPUT_DIRS = {"__pycache__", "cand", "ref", "correctness"}
# The integrator's A/B records, written into the candidate dir during and after the A/B
# (roles/e2e_integrator.md: CB=$EVAL_DIR/overlay/cand_<short>). Records, not overlay content.
_AB_RECORDS = {"integrate_result.json", "e2e_validation.json", "manifests.json"}
# What the server imports or loads. Only these must be unchanged since the A/B's candidate leg: the
# integrator may still write records next to them.
_CODE_SUFFIXES = (".py", ".so", ".hip", ".cu", ".cpp", ".cc", ".h", ".hpp", ".hsaco", ".co")
_ABS_PATH = re.compile(r"""["'](/[^"'\s]+)["']""")


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# The overlay's merged injection manifest (read by sitecustomize.py). Stacking a region on an accepted
# overlay must ADD its own entries here, so this file can never stay byte-identical; it is checked by key
# instead, with the same identity keys and target-overlap rule as overlay_setup.py merge.
_MANIFEST = "_overlay_manifest.json"
_MANIFEST_KEYS = {"modules": "module", "rebinds": "target", "markers": "target", "captures": "target",
                  "installers": "id"}


_OVERLAY_SETUP = None


def _targets_overlap(left: str, right: str) -> bool:
    """overlay_setup.py's rule: one target replaces (part of) the other ("pkg.mod" covers "pkg.mod:fn")."""
    global _OVERLAY_SETUP
    if _OVERLAY_SETUP is None:
        spec = importlib.util.spec_from_file_location("_geak_overlay_setup",
                                                      Path(__file__).with_name("overlay_setup.py"))
        _OVERLAY_SETUP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OVERLAY_SETUP)
    return _OVERLAY_SETUP._targets_overlap(left, right)


def _sites(kind: str, entry: dict) -> list[str]:
    """The serving sites an entry replaces: a module or rebind target, an installer's declared targets."""
    if kind == "installers":
        return [t for t in entry.get("targets") or [] if isinstance(t, str)]
    key = entry.get(_MANIFEST_KEYS[kind])
    return [key] if kind in ("modules", "rebinds") and isinstance(key, str) else []


_SCOPES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try) + (
    (ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _scope(body: list) -> list:
    """Every statement of one scope, including those nested in if / for / while / with / try blocks (which
    run in the same scope), but not inside a function or class body."""
    out = []
    for n in body:
        out.append(n)
        if isinstance(n, _SCOPES):
            for field in ("body", "orelse", "finalbody"):
                out += _scope(getattr(n, field, []) or [])
            for h in getattr(n, "handlers", []) or []:
                out += _scope(h.body)
    return out


def _names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [x for e in target.elts for x in _names(e)]
    return []


def _binds(n, name: str) -> bool:
    """Whether statement n binds name in its own scope."""
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return n.name == name
    if isinstance(n, ast.Assign):
        return any(name in _names(t) for t in n.targets)
    if isinstance(n, (ast.AnnAssign, ast.AugAssign)):
        return name in _names(n.target)
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        return any((al.asname or al.name.split(".")[0]) == name for al in n.names)
    return False


def _dotted(e) -> str | None:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        base = _dotted(e.value)
        return base and f"{base}.{e.attr}"
    return None


def _attr_writes(body: list, attr: str) -> list:
    """Module-scope statements that write the dotted attribute from outside its class: C.m = ... (+= / :),
    setattr(C, "m", ...) and delattr(C, "m")."""
    owner, _, last = attr.rpartition(".")
    out = []
    for n in _scope(body):
        targets = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign)) else [])
        if any(_dotted(t) == attr for t in targets):
            out.append(n)
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
              and n.value.func.id in ("setattr", "delattr") and len(n.value.args) >= 2
              and _dotted(n.value.args[0]) == owner and isinstance(n.value.args[1], ast.Constant)
              and n.value.args[1].value == last):
            out.append(n)
    return out


def _definition(source: str, attr: str) -> list[str] | None:
    """What decides attr (a name or Class.member) at import time, as ast dumps: every binding of each part in
    its scope (nested if / try / ... blocks and imports included) and, for a member, every module-scope write
    of it from outside the class. None when attr is not bound."""
    mod = ast.parse(source)
    parts, body, sig = attr.split("."), mod.body, []
    for i, part in enumerate(parts):
        nodes = [n for n in _scope(body) if _binds(n, part)]
        if not nodes:
            return None
        last = nodes[-1]
        if i == len(parts) - 1:
            sig += [ast.dump(n, include_attributes=False) for n in nodes]
        elif isinstance(last, ast.ClassDef) and len(nodes) == 1:
            body = last.body
        else:   # the class is rebound or redefined: compare every binding of it, member lookup is undecidable
            return sig + [ast.dump(n, include_attributes=False) for n in nodes]
    if len(parts) > 1:
        sig += [ast.dump(n, include_attributes=False) for n in _attr_writes(mod.body, attr)]
    return sig


def _installed_source(module: str, exclude: list[Path]) -> Path | None:
    """The installed file of a dotted module, found WITHOUT importing it or its parents: the top-level
    package's spec (importlib.util.find_spec(<top>) goes through every sys.meta_path finder, including a
    PEP 660 editable-install finder, and imports nothing for a top-level name), then the remaining parts under
    its submodule_search_locations; else a plain sys.path scan. Overlay dirs are skipped."""
    parts = module.split(".")
    skip = lambda p: any(p == x or x in p.parents for x in exclude)

    def under(bases):
        for base in bases:
            base = Path(base).resolve()
            if skip(base):
                continue
            for cand in (base.joinpath(*parts[1:]).with_suffix(".py") if parts[1:] else None,
                         base.joinpath(*parts[1:], "__init__.py")):
                if cand is not None and cand.is_file():
                    return cand
        return None

    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None:
        if spec.submodule_search_locations:
            hit = under(list(spec.submodule_search_locations))
            if hit:
                return hit
        elif len(parts) == 1 and spec.origin and Path(spec.origin).is_file() and not skip(Path(spec.origin).resolve()):
            return Path(spec.origin)
    for entry in sys.path:
        base = Path(entry or ".").resolve()
        if skip(base):
            continue
        for cand in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
            if cand.is_file():
                return cand
    return None


def _unchanged_rebind_site(target: str, module_entry: dict, root: Path, exclude: list[Path]) -> str:
    """'' when a rebind of target replaces only code the initial patched module left as installed, else why not."""
    module, _, attr = target.partition(":")
    if not attr or module != module_entry.get("module"):
        return "it is not an attribute of the replaced module itself"
    stock = _installed_source(module, exclude)
    if stock is None:
        return f"the installed {module} source was not found on sys.path"
    try:
        mine = _definition((root / str(module_entry.get("file") or "")).read_text(encoding="utf-8"), attr)
        theirs = _definition(stock.read_text(encoding="utf-8"), attr)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
        return f"cannot compare {attr}: {type(e).__name__}"
    if mine is None or theirs is None:
        return f"{attr} is not defined in {'the patched' if mine is None else 'the installed'} {module}"
    return "" if mine == theirs else f"the initial overlay changed {attr} in {module}"


def _manifest_superset_problems(new_p: Path, old_p: Path, root: Path) -> list[str]:
    """Problems if the new manifest drops, alters, duplicates or overrides an entry of the initial one."""
    old, new = _read_json(old_p), _read_json(new_p)
    if old is None:
        return [f"initial {_MANIFEST} ({root}) is not a JSON object"]
    if new is None:
        return [f"{_MANIFEST} is not a JSON object"]
    out: list[str] = []
    exclude = [root.resolve(), new_p.parent.resolve()]
    for key, val in old.items():
        if key not in _MANIFEST_KEYS and new.get(key) != val:
            out.append(f"{_MANIFEST}: initial field '{key}' ({root}) was changed")
    held: list[tuple[str, str, str, dict]] = []   # (site, kind, key, entry) of every initial entry
    for kind, ident in _MANIFEST_KEYS.items():
        before, after = old.get(kind, []), new.get(kind, [])
        if not isinstance(before, list):
            out.append(f"initial {_MANIFEST} ({root}): '{kind}' is not a list")
            continue
        if not isinstance(after, list):
            out.append(f"{_MANIFEST}: '{kind}' is not a list")
            continue
        for e in before:
            k = e.get(ident) if isinstance(e, dict) else None
            same = [x for x in after if isinstance(x, dict) and x.get(ident) == k]
            if e not in after:
                out.append(f"{_MANIFEST}: initial '{kind}' entry dropped or altered ({root}): {json.dumps(e)[:200]}")
            elif len(same) > 1:
                out.append(f"{_MANIFEST}: initial '{kind}' entry {k!r} ({root}) has another entry with the same "
                           f"{ident}, which replaces it at load time")
            if isinstance(e, dict):
                held += [(s, kind, k, e) for s in _sites(kind, e)]
    for kind, ident in _MANIFEST_KEYS.items():
        after = new.get(kind, [])
        for e in after if isinstance(after, list) else []:
            if not isinstance(e, dict) or e in (old.get(kind) or []):
                continue
            for site in _sites(kind, e):
                for prev, pkind, pkey, pentry in held:
                    if not _targets_overlap(site, prev):
                        continue
                    why = (_unchanged_rebind_site(site, pentry, root, exclude) if kind == "rebinds" and pkind == "modules"
                           else "")
                    if kind == "rebinds" and pkind == "modules" and not why:
                        continue
                    out.append(f"{_MANIFEST}: new '{kind}' entry {e.get(ident)!r} targets {site}, which overlaps "
                               f"{prev} of the initial {pkind}:{pkey} ({root})" + (f": {why}" if why else ""))
    return out


def _overlay_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _OVERLAY_OUTPUT_DIRS]
        for f in filenames:
            if f.endswith((".pyc", ".log", ".out")) or f in _AB_RECORDS:
                continue
            p = Path(dirpath) / f
            out[str(p.relative_to(root))] = p
    return out


def _ab_record(overlay: Path, run_dir: Path) -> tuple[dict | None, Path | None]:
    """The integrate_result.json whose A/B accepted exactly this overlay directory: the overlay's own
    record first, then any other under the run dir."""
    own = overlay / "integrate_result.json"
    for p in ([own] if own.is_file() else []) + sorted(run_dir.rglob("integrate_result.json")):
        d = _read_json(p) or {}
        acc = d.get("accepted_overlay")
        if acc and Path(acc).resolve() == overlay:
            return d, p
    return None, None


def validate_overlay(kf_overlay: str, initial_overlay: str, run_dir: Path) -> dict:
    """The checks in the module docstring. Returns {"ok", "problems", "manifest_sha256", "files"}."""
    problems: list[str] = []
    run_dir = Path(run_dir).resolve()
    overlay = Path(kf_overlay).resolve() if kf_overlay else Path()
    if not kf_overlay or not overlay.is_dir():
        return {"ok": False, "problems": [f"overlay {kf_overlay!r} is not a directory"]}
    if run_dir not in overlay.parents:
        problems.append(f"overlay {overlay} is outside the run dir {run_dir}")
    files = _overlay_files(overlay)
    site = overlay / "sitecustomize.py"
    if not site.is_file():
        problems.append("no sitecustomize.py at the overlay root")
    for rel, p in sorted(files.items()):
        if not rel.endswith(".py"):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            compile(src, str(p), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
            problems.append(f"{rel} does not compile: {type(e).__name__}: {e}"[:300])
            continue
        for ref in _ABS_PATH.findall(src):
            if ref.startswith(str(run_dir.parent) + os.sep) and not Path(ref).exists():
                problems.append(f"{rel} names {ref}, which does not exist")
    site_src = site.read_text(encoding="utf-8", errors="replace") if site.is_file() else ""
    for root in [Path(x) for x in str(initial_overlay or "").split(os.pathsep) if x.strip()]:
        if not root.is_dir():
            continue
        for rel, p in sorted(_overlay_files(root).items()):
            want = p.read_bytes()
            if rel == "sitecustomize.py":
                if site.is_file() and site.read_bytes() == want:
                    continue
                kept = [r for r, q in files.items() if r != "sitecustomize.py" and q.read_bytes() == want]
                imported = [r for r in kept if re.search(
                    r"\b" + re.escape(Path(r).with_suffix("").as_posix().replace("/", ".")) + r"\b", site_src)]
                if not kept:
                    problems.append(f"the initial overlay's sitecustomize.py ({root}) is not kept")
                elif not imported:
                    problems.append(f"the initial overlay's sitecustomize.py is kept as {kept[0]} "
                                    "but the new sitecustomize.py does not import it")
            elif rel not in files:
                problems.append(f"initial overlay file {rel} ({root}) is missing")
            elif rel == _MANIFEST:
                problems.extend(_manifest_superset_problems(files[rel], p, root))
            elif files[rel].read_bytes() != want:
                problems.append(f"initial overlay file {rel} ({root}) was changed")
    rec, rec_path = _ab_record(overlay, run_dir)
    if rec is None:
        problems.append("no integrate_result.json accepted this overlay directory")
    elif not rec.get("ab_complete"):
        problems.append(f"{rec_path}: the A/B is not complete")
    else:
        leg = Path(((rec.get("measurement_evidence") or {}).get("cand_summary")) or "")
        stamp = (leg if leg.is_file() else rec_path).stat().st_mtime
        changed = [r for r, p in files.items() if r.endswith(_CODE_SUFFIXES) and p.stat().st_mtime > stamp + 2]
        if changed:
            problems.append(f"overlay sources changed after the A/B measured it: {', '.join(sorted(changed)[:5])}")
    h = hashlib.sha256()
    for rel, p in sorted(files.items()):
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return {"ok": not problems, "problems": problems, "manifest_sha256": h.hexdigest(), "files": len(files)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overlay", required=True, help="the accepted overlay dir")
    ap.add_argument("--initial", default="", help="the overlay(s) it was built on (os.pathsep-separated)")
    ap.add_argument("--root", required=True, help="the run's EVAL_DIR")
    a = ap.parse_args(argv)
    try:
        out = validate_overlay(a.overlay, a.initial, Path(a.root))
    except Exception as e:  # a check that cannot run is a failed check, never an accept
        out = {"ok": False, "problems": [f"check failed: {type(e).__name__}: {e}"[:500]]}
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
