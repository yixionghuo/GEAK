"""fusion_overlay_check.py: the accepted KernelFusion overlay keeps the overlay it was built on and is the one its A/B measured."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPTS_DIR / "fusion_overlay_check.py"
_spec = importlib.util.spec_from_file_location("fusion_overlay_check", SCRIPT)
foc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(foc)


def make_overlay(run_dir, initial=None, site="import _kf_c1\n", extra=None):
    """An accepted overlay as the integrator leaves it: sources + the integrate_result.json of its A/B."""
    ov = Path(run_dir) / "overlay" / "cand_kf_c1"
    ov.mkdir(parents=True, exist_ok=True)
    if initial:
        for rel, text in initial.items():
            (ov / rel).parent.mkdir(parents=True, exist_ok=True)
            (ov / rel).write_text(text)
    (ov / "_kf_c1.py").write_text("X = 1\n")
    (ov / "sitecustomize.py").write_text(site)
    for rel, text in (extra or {}).items():
        (ov / rel).write_text(text)
    (ov / "cand").mkdir(exist_ok=True)
    (ov / "cand" / "bench_summary.json").write_text("{}")
    (Path(run_dir) / "integrate_kf_c1").mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "integrate_kf_c1" / "integrate_result.json").write_text(json.dumps(
        {"ab_complete": True, "accepted_overlay": str(ov),
         "measurement_evidence": {"cand_summary": str(ov / "cand" / "bench_summary.json")}}))
    return str(ov)


def _initial(tmp_path, files):
    d = tmp_path / "initial_overlay"
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return str(d)


def test_validate_overlay_accepts_an_overlay_that_keeps_the_initial_overlay(tmp_path):
    run_dir = tmp_path / "e2e_x"
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n", "pkg/a.py": "A = 1\n"})
    ov = make_overlay(run_dir, initial={"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n",
                                        "pkg/a.py": "A = 1\n"},
                      site="import _initial_sitecustomize\nimport _kf_c1\n")
    check = foc.validate_overlay(ov, init, run_dir)
    assert check["ok"], check["problems"]
    assert len(check["manifest_sha256"]) == 64


def test_validate_overlay_rejects_what_would_lose_or_break_an_optimization(tmp_path):
    init_files = {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    kept = {"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    good_site = "import _initial_sitecustomize\nimport _kf_c1\n"
    cases = {
        "missing": ({"initial": {"_initial_sitecustomize.py": "import tuned\n"}, "site": good_site}, "tuned.py"),
        "changed": ({"initial": {**kept, "tuned.py": "T = 3\n"}, "site": good_site}, "was changed"),
        "not imported": ({"initial": kept, "site": "import _kf_c1\n"}, "does not import it"),
        "dropped site": ({"initial": {"tuned.py": "T = 2\n"}, "site": good_site}, "is not kept"),
        "syntax": ({"initial": kept, "site": good_site, "extra": {"_kf_bad.py": "def (:\n"}}, "does not compile"),
        "dangling kernel": ({"initial": kept, "site": good_site,
                             "extra": {"_kf_k.py": "K = '{kf}/kernels/c1/kernel'\n"}}, "does not exist"),
    }
    for i, (name, (kw, needle)) in enumerate(cases.items()):
        root = tmp_path / str(i)
        run_dir = root / "e2e_x"
        if "extra" in kw:
            kw["extra"] = {k: v.replace("{kf}", str(run_dir)) for k, v in kw["extra"].items()}
        ov = make_overlay(run_dir, **kw)
        check = foc.validate_overlay(ov, _initial(root, init_files), run_dir)
        assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])


def test_validate_overlay_ties_the_overlay_to_the_ab_that_accepted_it(tmp_path):
    run_dir = tmp_path / "e2e_x"
    ov = make_overlay(run_dir)
    assert foc.validate_overlay(ov, "", run_dir)["ok"]
    first = foc.validate_overlay(ov, "", run_dir)["manifest_sha256"]
    # the integrator's A/B records land in the candidate dir after the A/B (DSv4.1-Flash c64: 4 min later):
    # they are records, not overlay content -- neither a change nor part of the manifest
    later = os.path.getmtime(Path(ov) / "cand" / "bench_summary.json") + 240
    for rec in ("e2e_validation.json", "manifests.json"):
        (Path(ov) / rec).write_text("{}")
        os.utime(Path(ov) / rec, (later, later))
    check = foc.validate_overlay(ov, "", run_dir)
    assert check["ok"], check["problems"]
    assert check["manifest_sha256"] == first
    os.utime(Path(ov) / "_kf_c1.py", (later, later))                       # code edited after the A/B
    assert any("after the A/B" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])
    (run_dir / "integrate_kf_c1" / "integrate_result.json").unlink()
    assert any("no integrate_result.json" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])
    elsewhere = make_overlay(tmp_path / "other_run")
    assert any("outside the run dir" in p for p in foc.validate_overlay(elsewhere, "", run_dir)["problems"])


def test_validate_overlay_edge_cases(tmp_path):
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"})
    # the overlay loads its kernels without touching sitecustomize.py; binaries are not compiled as Python;
    # an initial overlay that is gone has nothing left to keep; a corrupt record elsewhere is ignored
    run_dir = tmp_path / "e2e_a"
    ov = make_overlay(run_dir, initial={"tuned.py": "T = 2\n"}, site="import tuned\n",
                      extra={"_kf_c1.hsaco": "\x7fELF"})
    (run_dir / "integrate_aborted").mkdir()                                  # read before integrate_kf_c1
    (run_dir / "integrate_aborted" / "integrate_result.json").write_text("{not json")
    check = foc.validate_overlay(ov, os.pathsep.join([init, str(tmp_path / "gone")]), run_dir)
    assert check["ok"], check["problems"]
    assert check["files"] == 4                                                # tuned, sitecustomize, _kf_c1 .py + .hsaco

    run_dir = tmp_path / "e2e_b"
    ov = make_overlay(run_dir)
    (Path(ov) / "sitecustomize.py").unlink()
    assert "no sitecustomize.py at the overlay root" in foc.validate_overlay(ov, "", run_dir)["problems"]

    run_dir = tmp_path / "e2e_c"
    ov = make_overlay(run_dir)
    rec = run_dir / "integrate_kf_c1" / "integrate_result.json"
    rec.write_text(json.dumps({**json.loads(rec.read_text()), "ab_complete": False}))
    assert any("the A/B is not complete" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])


def _kept_overlay(tmp_path):
    run_dir = tmp_path / "e2e_x"
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"})
    ov = make_overlay(run_dir, initial={"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"},
                      site="import _initial_sitecustomize\nimport _kf_c1\n")
    return run_dir, init, ov


def test_main_prints_one_json_object_and_exits_1_when_the_check_fails(tmp_path, capsys, monkeypatch):
    run_dir, init, ov = _kept_overlay(tmp_path)

    def cli(*argv):
        rc = foc.main(list(argv))
        return rc, json.loads(capsys.readouterr().out)

    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert rc == 0 and out["ok"] and out["problems"] == []
    (Path(ov) / "tuned.py").write_text("T = 3\n")
    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert rc == 1 and not out["ok"] and any("was changed" in p for p in out["problems"])
    rc, out = cli("--overlay", str(tmp_path / "nope"), "--root", str(run_dir))
    assert rc == 1 and not out["ok"] and out["problems"]

    def broken(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(foc, "validate_overlay", broken)                      # a check that cannot run never accepts
    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert (rc, out) == (1, {"ok": False, "problems": ["check failed: OSError: disk gone"]})


def test_the_script_runs_as_the_agent_runs_it(tmp_path):
    run_dir, init, ov = _kept_overlay(tmp_path)
    p = subprocess.run([sys.executable, str(SCRIPT), "--overlay", ov, "--initial", init, "--root", str(run_dir)],
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0 and json.loads(p.stdout)["ok"], p.stderr


def test_validate_overlay_checks_the_merged_injection_manifest_as_a_superset(tmp_path):
    """The integrator's overlays share ONE sitecustomize.py that reads _overlay_manifest.json. Stacking a
    region on an accepted overlay must ADD its module to that manifest, so the manifest can never stay
    byte-identical: it is checked as a superset (DSv4.1-Flash TP2 c32: kf_hc_boundary_norm_sinkhorn on the
    accepted wo_a overlay was rejected by the old byte-identity rule although nothing of wo_a changed)."""
    site = "import json, os\n_MAN = os.path.join(os.path.dirname(__file__), '_overlay_manifest.json')\n"
    woa = {"module": "sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid",
           "file": "_patched/sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid.py"}
    kf = {"module": "sglang.srt.models.deepseek_v4_platform_stage",
          "file": "_patched/sglang.srt.models.deepseek_v4_platform_stage.py"}

    def manifest(modules, **extra):
        return json.dumps({"modules": modules, "rebinds": [], "markers": [], "captures": [], **extra})

    patched_woa = "W = 1\n"
    init_files = {"sitecustomize.py": site, "_overlay_manifest.json": manifest([woa]), woa["file"]: patched_woa}
    cases = {
        "superset": ({"_overlay_manifest.json": manifest([woa, kf]), woa["file"]: patched_woa,
                      kf["file"]: "K = 1\n"}, None),
        "dropped entry": ({"_overlay_manifest.json": manifest([kf]), woa["file"]: patched_woa,
                           kf["file"]: "K = 1\n"}, "dropped or altered"),
        "altered entry": ({"_overlay_manifest.json": manifest([{**woa, "file": kf["file"]}, kf]),
                           woa["file"]: patched_woa, kf["file"]: "K = 1\n"}, "dropped or altered"),
        "added field": ({"_overlay_manifest.json": manifest([woa, kf], version=2), woa["file"]: patched_woa,
                           kf["file"]: "K = 1\n"}, None),
        "changed patched module": ({"_overlay_manifest.json": manifest([woa, kf]), woa["file"]: "W = 2\n",
                                    kf["file"]: "K = 1\n"}, "was changed"),
        "not json": ({"_overlay_manifest.json": "[", woa["file"]: patched_woa}, "not a JSON object"),
    }
    for i, (name, (files, needle)) in enumerate(cases.items()):
        root = tmp_path / str(i)
        run_dir = root / "e2e_x"
        ov = make_overlay(run_dir, initial=files, site=site)
        check = foc.validate_overlay(ov, _initial(root, init_files), run_dir)
        if needle is None:
            assert check["ok"], (name, check["problems"])
        else:
            assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])
    # a field the initial manifest already had may not change
    root = tmp_path / "field"
    run_dir = root / "e2e_x"
    ov = make_overlay(run_dir, initial={"_overlay_manifest.json": manifest([woa, kf], version=2),
                                        woa["file"]: patched_woa}, site=site)
    init = _initial(root, {**init_files, "_overlay_manifest.json": manifest([woa], version=1)})
    check = foc.validate_overlay(ov, init, run_dir)
    assert not check["ok"] and any("'version'" in p for p in check["problems"]), check["problems"]


def _manifest_check(tmp_path, name, init_manifest, new_manifest, files=None):
    """validate_overlay on an overlay whose only manifest change is init_manifest -> new_manifest."""
    site = "import json, os\n"
    files = files or {}
    root = tmp_path / name
    run_dir = root / "e2e_x"
    ov = make_overlay(run_dir, initial={"_overlay_manifest.json": json.dumps(new_manifest), **files}, site=site)
    init = _initial(root, {"sitecustomize.py": site, "_overlay_manifest.json": json.dumps(init_manifest), **files})
    return foc.validate_overlay(ov, init, run_dir)


def test_validate_overlay_checks_the_manifest_by_key_like_overlay_merge(tmp_path):
    """The manifest is checked with overlay_setup.py merge's identity keys and target-overlap rule: a stacked
    region may add installers (they are list entries, not a field that must stay equal), but it may not keep an
    initial entry while adding another one with the same key (sitecustomize loads the later one), nor target a
    site an initial entry already replaces (a rebind runs after module injection and overrides the patch)."""
    empty = {"modules": [], "rebinds": [], "markers": [], "captures": []}
    mod = {"module": "pkg.a", "file": "_patched/pkg.a.py"}
    files = {"_patched/pkg.a.py": "A = 1\n", "_patched/pkg.a2.py": "A = 2\n"}

    def inst(i, targets=()):
        return {"id": i, "module": f"inst_{i}", "callable": "install", "required": True,
                "targets": list(targets), "kernels": []}

    rb = {"target": "pkg.b:fn", "impl_module": "kf_impl", "impl_attr": "fn"}
    cases = {
        "installer stacked on installer": ({**empty, "installers": [inst("a", ["pkg.c:f"])]},
                                           {**empty, "installers": [inst("a", ["pkg.c:f"]), inst("b", ["pkg.d:g"])]},
                                           None),
        "entries reordered": ({**empty, "rebinds": [rb, {**rb, "target": "pkg.b:gn"}]},
                              {**empty, "rebinds": [{**rb, "target": "pkg.b:gn"}, rb]}, None),
        "installer dropped": ({**empty, "installers": [inst("a")]}, {**empty, "installers": []}, "dropped or altered"),
        "same module twice": ({**empty, "modules": [mod]},
                              {**empty, "modules": [mod, {"module": "pkg.a", "file": "_patched/pkg.a2.py"}]},
                              "same module"),
        "rebind inside a replaced module": ({**empty, "modules": [mod]},
                                            {**empty, "modules": [mod], "rebinds": [{**rb, "target": "pkg.a:fn"}]},
                                            "overlaps pkg.a"),
        "module over a rebind": ({**empty, "rebinds": [rb]},
                                 {**empty, "rebinds": [rb], "modules": [{"module": "pkg.b", "file": "_patched/pkg.a2.py"}]},
                                 "overlaps pkg.b:fn"),
        "installer target over a rebind": ({**empty, "rebinds": [rb]},
                                           {**empty, "rebinds": [rb], "installers": [inst("b", ["pkg.b:fn"])]},
                                           "overlaps pkg.b:fn"),
        "list on one side only": ({**empty, "rebinds": [rb]}, {**empty, "rebinds": {"x": 1}}, "'rebinds' is not a list"),
    }
    for name, (before, after, needle) in cases.items():
        check = _manifest_check(tmp_path, name.replace(" ", "_"), before, after, files)
        if needle is None:
            assert check["ok"], (name, check["problems"])
        else:
            assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])


def test_a_rebind_on_an_unchanged_function_of_a_replaced_module_may_stack(tmp_path, monkeypatch):
    """The one allowed overlap: the initial overlay replaces module pkg.a, a stacked region rebinds pkg.a:X, and
    X's definition in the patched file is AST-identical to the installed pkg.a's (DSv4.1: the draft rebind of
    _forward_hc_pre_from_prev on the hc-patched module, whose copy of that function is the stock one). Any change
    to X, a missing X, a rebind of another module, or an installed source that cannot be found stays a conflict."""
    stock_src = ("def fn(x):\n    return x + 1\n\n\ndef helper(x):\n    return x * 2\n\n\n"
                 "class C:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n")
    # The patched module changes helper and C.n only; fn and C.m keep their stock definitions (comments and
    # blank lines differ, which an AST comparison ignores).
    patched_src = ("# patched by the hc region\ndef fn(x):\n    # same body\n    return x + 1\n\n"
                   "def helper(x):\n    return x << 1\n\n"
                   "class C:\n    def m(self):\n        return 1\n    def n(self):\n        return 3\n")
    stock = tmp_path / "site_packages"
    (stock / "pkg").mkdir(parents=True)
    (stock / "pkg" / "__init__.py").write_text("")
    (stock / "pkg" / "a.py").write_text(stock_src)
    monkeypatch.syspath_prepend(str(stock))
    empty = {"modules": [], "rebinds": [], "markers": [], "captures": []}
    mod = {"module": "pkg.a", "file": "_patched/pkg.a.py"}
    files = {"_patched/pkg.a.py": patched_src}

    def stacked(target):
        return ({**empty, "modules": [mod]},
                {**empty, "modules": [mod], "rebinds": [{"target": target, "impl_module": "kf_impl", "impl_attr": "f"}]})

    cases = {
        "unchanged function": ("pkg.a:fn", None),
        "unchanged method": ("pkg.a:C.m", None),
        "changed function": ("pkg.a:helper", "the initial overlay changed helper"),
        "changed method": ("pkg.a:C.n", "the initial overlay changed C.n"),
        "missing attribute": ("pkg.a:nope", "nope is not defined in the patched pkg.a"),
        "whole module rebind": ("pkg.a", "overlaps pkg.a"),
    }
    for name, (target, needle) in cases.items():
        before, after = stacked(target)
        check = _manifest_check(tmp_path, name.replace(" ", "_"), before, after, files)
        if needle is None:
            assert check["ok"], (name, check["problems"])
        else:
            assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])

    # fail closed: without the installed source on sys.path the same unchanged function is a conflict again
    monkeypatch.syspath_prepend(str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(stock)])
    before, after = stacked("pkg.a:fn")
    check = _manifest_check(tmp_path, "no_stock", before, after, files)
    assert not check["ok"] and any("was not found on sys.path" in p for p in check["problems"]), check["problems"]


def test_the_installed_source_is_found_through_an_editable_install_finder(tmp_path, monkeypatch):
    """A PEP 660 editable install (sglang in the serving images) is resolved by a sys.meta_path finder, not a
    sys.path entry. The lookup goes through the top-level package's spec, so it finds the source there -- and
    imports nothing (the package stays out of sys.modules)."""
    import importlib.abc
    import importlib.machinery
    src = tmp_path / "workspace" / "edpkg"
    (src / "sub").mkdir(parents=True)
    (src / "__init__.py").write_text("raise RuntimeError('importing edpkg is not allowed here')\n")
    (src / "sub" / "__init__.py").write_text("")
    (src / "sub" / "mod.py").write_text("def fn(x):\n    return x\n\n\ndef g(x):\n    return 1\n")

    class EditableFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "edpkg":
                return None
            return importlib.util.spec_from_file_location(name, src / "__init__.py",
                                                          submodule_search_locations=[str(src)])

    monkeypatch.setattr(sys, "meta_path", sys.meta_path + [EditableFinder()])
    assert foc._installed_source("edpkg.sub.mod", []) == src / "sub" / "mod.py"
    assert "edpkg" not in sys.modules
    empty = {"modules": [], "rebinds": [], "markers": [], "captures": []}
    mod = {"module": "edpkg.sub.mod", "file": "_patched/edpkg.sub.mod.py"}
    patched = "def fn(x):\n    return x\n\n\ndef g(x):\n    return 2\n"
    for target, ok in (("edpkg.sub.mod:fn", True), ("edpkg.sub.mod:g", False)):
        check = _manifest_check(tmp_path, target.replace(":", "_"), {**empty, "modules": [mod]},
                                {**empty, "modules": [mod],
                                 "rebinds": [{"target": target, "impl_module": "kf_impl", "impl_attr": "f"}]},
                                {"_patched/edpkg.sub.mod.py": patched})
        assert check["ok"] is ok, (target, check["problems"])
    assert "edpkg" not in sys.modules


def test_a_definition_counts_every_binding_that_decides_it_at_import_time(tmp_path, monkeypatch):
    """The patched module can change a function without touching its def: a module-level write of the member
    (C.m = ..., setattr(C, "m", ...)), a redefinition inside an if / try block, or an import that rebinds the
    name. Each of these is a change, so the rebind on it stays a conflict."""
    stock_src = "class C:\n    def m(self):\n        return 1\n\n\ndef fn(x):\n    return x\n"
    stock = tmp_path / "site_packages"
    (stock / "pkg2").mkdir(parents=True)
    (stock / "pkg2" / "__init__.py").write_text("")
    (stock / "pkg2" / "b.py").write_text(stock_src)
    monkeypatch.syspath_prepend(str(stock))
    empty = {"modules": [], "rebinds": [], "markers": [], "captures": []}
    mod = {"module": "pkg2.b", "file": "_patched/pkg2.b.py"}
    variants = {
        "class attribute assigned at module level": ("pkg2.b:C.m", stock_src + "\n\ndef _m2(self):\n    return 2\n\n\nC.m = _m2\n"),
        "setattr at module level": ("pkg2.b:C.m", stock_src + "\nsetattr(C, 'm', lambda self: 2)\n"),
        "redefinition in an if block": ("pkg2.b:fn", stock_src + "\nif True:\n    def fn(x):\n        return -x\n"),
        "rebinding import": ("pkg2.b:fn", stock_src + "\nfrom os.path import basename as fn\n"),
        "class redefined": ("pkg2.b:C.m", stock_src + "\nclass C:\n    def m(self):\n        return 1\n"),
    }
    for name, (target, patched) in variants.items():
        check = _manifest_check(tmp_path, name.replace(" ", "_"), {**empty, "modules": [mod]},
                                {**empty, "modules": [mod],
                                 "rebinds": [{"target": target, "impl_module": "kf_impl", "impl_attr": "f"}]},
                                {"_patched/pkg2.b.py": patched})
        assert not check["ok"] and any("the initial overlay changed" in p for p in check["problems"]), \
            (name, check["problems"])
    # the same statements in BOTH the installed and the patched module are not a change
    same = stock_src + "\nsetattr(C, 'm', lambda self: 2)\n"
    (stock / "pkg2" / "b.py").write_text(same)
    check = _manifest_check(tmp_path, "same_in_both", {**empty, "modules": [mod]},
                            {**empty, "modules": [mod],
                             "rebinds": [{"target": "pkg2.b:C.m", "impl_module": "kf_impl", "impl_attr": "f"}]},
                            {"_patched/pkg2.b.py": same})
    assert check["ok"], check["problems"]
