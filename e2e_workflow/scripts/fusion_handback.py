#!/usr/bin/env python3
"""The hand-back of a fusion region: one directory, whoever wrote the kernel.

Every implementer of a fusion region (a GEAK kernel lane following the fusion-kernel skill, or an outside
team) hands back the same directory, and KernelFusion accepts it the same way:

    <handback>/
      kernel_src/      the implementation (the only thing that changed against the region UT's task dir)
      support.json     where it engages: {"phases": [...], "m_range": "...", "dtypes": [...], "refusals": [...]},
                       and for a multi-rank UT (meta.mega / meta.required_gpu_count > 1) "required_gpu_count"
      init.py          optional: server start-up preparation (JIT, registrations)
      ut_result.json   the region UT's result on this implementation
      notes.md         the verdict (all cases at the floor / some, the rest routed back / not feasible), the design

  assemble  builds it from a kernel lane's result: applies the lane's final_patch.diff to a copy of the task
            dir's kernel_src/, moves the files the lane wrote under kernel_src/_handback/ to the top, and
            records the lane's own validation (its director re-ran the same UT on a fresh copy) as ut_result.json.
  check     checks a hand-back against its task dir: the layout, support.json, that every Python file
            compiles and the candidate_bind file exists, the UT result (passing, and at least
            meta.acceptance.min_speedup when the UT sets one). With --run it re-runs the task's own unittest
            on the hand-back (exit 0 = correct) and records that as ut_result.kf.json.

Both print one JSON object ({"ok", "problems", "warnings", "handback", ...}); a non-zero exit means "not ok".
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HB_DIR = "_handback"            # where a kernel lane writes support.json / notes.md / init.py inside kernel_src/
HB_FILES = ("support.json", "notes.md", "init.py", "ut_result.json")
FAILED = {"error", "fail", "failed", "rejected", "unknown", "incomplete", ""}
SPEEDUP_RE = re.compile(r"^(GEAK_WEIGHTED_SPEEDUP|GEAK_GEOMEAN_SPEEDUP)[=:]\s*([0-9.]+)", re.M)


def _load(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def _speedup(ut):
    for key in ("final_weighted", "weighted_speedup", "final_speedup", "speedup", "final_geomean"):
        v = (ut or {}).get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 0.0


def _ranks(d):
    """The GPUs one run of a multi-rank (L3 MegaKernel) UT or hand-back takes; 1 otherwise."""
    try:
        n = int((d or {}).get("required_gpu_count") or 0)
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else (2 if (d or {}).get("mega") else 1)


def assemble(task_dir, final_patch, lane_result, out, region=None):
    problems, warnings = [], []
    src = os.path.join(task_dir, "kernel_src")
    if not os.path.isdir(src):
        return {"ok": False, "problems": [f"task dir has no kernel_src/: {task_dir}"], "warnings": [], "handback": out}
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="hb_", dir=out)
    try:
        shutil.copytree(src, os.path.join(work, "kernel_src"), symlinks=True)
        if final_patch:
            p = subprocess.run(["git", "apply", "-p1", "--include=kernel_src/*", os.path.abspath(final_patch)],
                               cwd=work, capture_output=True, text=True)
            if p.returncode != 0:
                problems.append(f"final_patch does not apply to the task's kernel_src/: {p.stderr.strip()[:400]}")
        else:
            warnings.append("no final_patch: the hand-back is the task's kernel_src/ as it is")
        dst = os.path.join(out, "kernel_src")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(work, "kernel_src"), dst, symlinks=True)
        hb = os.path.join(dst, HB_DIR)
        if os.path.isdir(hb):
            for name in os.listdir(hb):
                shutil.move(os.path.join(hb, name), os.path.join(out, name))
            shutil.rmtree(hb)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    lane = lane_result or {}
    if region is not None and not os.path.exists(os.path.join(out, "support.json")):
        # A lane that did not state its envelope: the region's phases and rows, marked as derived. The guard in
        # kernel_src/ is what the server runs; an outside hand-back without support.json is still rejected.
        _dump(os.path.join(out, "support.json"), {
            "phases": list(region.get("phases") or []) or ["the region UT's cases"],
            "m_range": region.get("m_range") or "the region UT's cases", "dtypes": [], "refusals": [], "derived": True,
            **({"required_gpu_count": region["required_gpu_count"]} if region.get("required_gpu_count") else {})})
        warnings.append("support.json was not written by the implementer: derived from the region (phases, m_range)")
    if not os.path.exists(os.path.join(out, "ut_result.json")):
        _dump(os.path.join(out, "ut_result.json"), {
            "source": "kernel_lane", "note": "the lane's director re-ran the region UT on a fresh copy of the task dir",
            "validation_status": lane.get("validation_status", ""), "timing_basis": lane.get("timing_basis", ""),
            "final_speedup": lane.get("final_speedup"), "final_weighted": lane.get("final_weighted"),
            "final_geomean": lane.get("final_geomean"), "report_path": lane.get("report_path", "")})
    if not os.path.exists(os.path.join(out, "notes.md")):
        text = ""
        rp = lane.get("report_path") or ""
        if rp and os.path.isfile(rp):
            with open(rp, errors="replace") as fh:
                text = fh.read()
        with open(os.path.join(out, "notes.md"), "w") as fh:
            fh.write(text or f"# Hand-back\n\nAssembled from a kernel lane result: {json.dumps(lane)[:2000]}\n")
        warnings.append("notes.md was not written by the implementer: taken from the lane report")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "handback": out}


def _candidate_file(meta):
    bind = (meta or {}).get("candidate_bind") or {}
    return bind.get("file") or ""


def _run_unittest(task_dir, handback, gpu, timeout):
    """Re-run the task's own unittest with kernel_src/ replaced by the hand-back's (everything else linked)."""
    tmp = tempfile.mkdtemp(prefix="hb_run_")
    try:
        for name in os.listdir(task_dir):
            if name in ("kernel_src", "_cand_overlay"):
                continue
            os.symlink(os.path.join(os.path.abspath(task_dir), name), os.path.join(tmp, name))
        shutil.copytree(os.path.join(handback, "kernel_src"), os.path.join(tmp, "kernel_src"), symlinks=True)
        cmd = [sys.executable, "unittest.py"]
        lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernel_workflow", "scripts", "gpu_lock.sh")
        if gpu != "" and os.path.isfile(lock):
            cmd = ["bash", lock, str(gpu)] + cmd
        try:
            p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
            out, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            out, code = "", -1
        m = SPEEDUP_RE.findall(out)
        return {"exit_code": code, "correctness": "pass" if code == 0 else "fail",
                "speedup": float(m[-1][1]) if m else 0.0, "log_tail": out[-3000:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check(task_dir, handback, run=False, gpu="", timeout=7200):
    problems, warnings = [], []
    meta = _load(os.path.join(task_dir, "meta.json"), {})
    if not meta:
        problems.append(f"task dir has no readable meta.json: {task_dir}")
    ks = os.path.join(handback, "kernel_src")
    if not os.path.isdir(ks) or not os.listdir(ks):
        problems.append("kernel_src/ is missing or empty")
    else:
        for root, _, files in os.walk(ks):
            for f in files:
                if f.endswith(".py"):
                    try:
                        with open(os.path.join(root, f), "rb") as fh:
                            compile(fh.read(), os.path.join(root, f), "exec")
                    except (SyntaxError, ValueError) as e:
                        problems.append(f"{os.path.relpath(os.path.join(root, f), handback)} does not compile: {str(e)[:300]}")
        cf = _candidate_file(meta)
        if cf and not os.path.isfile(os.path.join(handback, cf)):
            problems.append(f"candidate_bind.file {cf} is not in the hand-back")
    sup = _load(os.path.join(handback, "support.json"))
    if not isinstance(sup, dict):
        problems.append("support.json is missing or not a JSON object")
    else:
        if not isinstance(sup.get("phases"), list) or not sup.get("phases"):
            problems.append("support.json: phases must be a non-empty list")
        if not isinstance(sup.get("refusals"), list):
            problems.append("support.json: refusals must be a list (empty when it engages everywhere)")
        if not (sup.get("m_range") or sup.get("widths")):
            problems.append("support.json: give m_range or widths (where the implementation engages)")
        need = _ranks(meta)
        if need > 1 and _ranks(sup) != need:
            problems.append(f"support.json: required_gpu_count must be {need}, the ranks the multi-rank UT runs "
                            f"(got {sup.get('required_gpu_count')})")
    notes = os.path.join(handback, "notes.md")
    if not os.path.isfile(notes) or os.path.getsize(notes) == 0:
        problems.append("notes.md is missing or empty")
    ut = _load(os.path.join(handback, "ut_result.json"))
    sp = _speedup(ut)
    min_sp = float(((meta or {}).get("acceptance") or {}).get("min_speedup") or 0)
    if not isinstance(ut, dict):
        problems.append("ut_result.json is missing or not a JSON object")
    else:
        status = str(ut.get("correctness") or ut.get("validation_status") or "").lower()
        if status in FAILED:
            problems.append(f"ut_result.json does not report a passing UT (status {status or 'none'})")
        elif status == "flagged":
            warnings.append("the lane's director flagged its validation (could not fully re-verify it)")
        if not sp:
            problems.append("ut_result.json has no speedup")
        elif min_sp and sp < min_sp:
            problems.append(f"speedup {sp:.3f} is below meta.acceptance.min_speedup {min_sp}")
    if not os.path.isfile(os.path.join(task_dir, "controls", "controls.json")):
        warnings.append("the task dir has no controls/controls.json (UT built before the unified standard)")
    res = {"handback": handback, "speedup": sp}
    if run and not problems:
        kf = _run_unittest(task_dir, handback, gpu, timeout)
        _dump(os.path.join(handback, "ut_result.kf.json"), kf)
        res["rerun"] = {k: kf[k] for k in ("exit_code", "correctness", "speedup")}
        if kf["exit_code"] != 0:
            problems.append(f"re-running the task's unittest on the hand-back failed (exit {kf['exit_code']})")
        elif sp and kf["speedup"] and not (0.85 <= kf["speedup"] / sp <= 1.15):
            warnings.append(f"re-run speedup {kf['speedup']:.3f} differs from ut_result.json's {sp:.3f} by more than 15%")
    res.update({"ok": not problems, "problems": problems, "warnings": warnings})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--task-dir", required=True)
    a.add_argument("--final-patch", default="")
    a.add_argument("--lane-result", default="{}", help="the kernel lane's return, as JSON")
    a.add_argument("--out", required=True)
    a.add_argument("--region", default="", help="the region's facts as JSON (phases, m_range, required_gpu_count): a lane that wrote "
                   "no support.json gets one derived from them")
    a.add_argument("--check", action="store_true", help="check the assembled hand-back too")
    c = sub.add_parser("check")
    c.add_argument("--task-dir", required=True)
    c.add_argument("--handback", required=True)
    c.add_argument("--run", action="store_true", help="re-run the task's unittest on the hand-back")
    c.add_argument("--gpu", default="")
    c.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args(argv)
    if args.cmd == "assemble":
        try:
            lane = json.loads(args.lane_result or "{}")
        except ValueError:
            lane = {}
        try:
            region = json.loads(args.region) if args.region else None
        except ValueError:
            region = None
        res = assemble(args.task_dir, args.final_patch, lane, args.out, region)
        if args.check and res["ok"]:
            chk = check(args.task_dir, args.out)
            res = {**chk, "warnings": res["warnings"] + chk["warnings"]}
    else:
        res = check(args.task_dir, args.handback, args.run, args.gpu, args.timeout)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
