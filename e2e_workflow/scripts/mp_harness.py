#!/usr/bin/env python3
"""Multi-rank harness of a fusion region UT (L3 MegaKernel, or any seam that spans ranks).

One process per rank (torch.multiprocessing), a process group whose initialization and collectives are
bounded by --timeout-s, and the same cases on both legs. Only the seam's outputs are compared (the max error
of each rank, then the max over ranks), and each case is timed per rank and reduced to the max over ranks
(the slowest rank sets the step). It prints the unified UT protocol and writes a result JSON:

    CORRECTNESS: PASS|FAIL max_err=<e> tol=<t> per_rank=[...]
    case <sig>: <baseline_ms> ms / <candidate_ms> ms / <speedup>x
    GEAK_GEOMEAN_SPEEDUP=<x>
    GEAK_WEIGHTED_SPEEDUP=<x>          (when cases carry weights)

Exit 0 only when every rank of every case is within tolerance. The cases module (``--cases``, a .py file in
the task) defines, per rank:

    CASES = [{"sig": "decode_bs8", "weight": 0.7}, ...]       (or cases() returning that list)
    TOL = 1e-3                                                (optional; --tol overrides)
    make_inputs(case, ctx) -> inputs                          (deterministic; distinct per rank)
    seam(leg, case, inputs, ctx) -> outputs                   (leg: "baseline" | "candidate"; a tensor or a
                                                               tuple / list / dict of tensors)

``ctx`` has ``rank``, ``world_size``, ``device`` and ``group``. ``seam`` is the region's seam call; the cases
module binds each leg (the baseline overlay, or the candidate's kernel_src/) the way the task's leg runner does.
A negative control is a cases module (or a --leg override) whose candidate is broken: it must exit 1.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import socket
import sys
import tempfile
import time


def _load(path):
    spec = importlib.util.spec_from_file_location(f"mp_cases_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases(mod):
    cs = mod.cases() if callable(getattr(mod, "cases", None)) else getattr(mod, "CASES", [])
    return [c if isinstance(c, dict) else {"sig": str(c)} for c in cs]


def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, dict):
        return [t for k in sorted(x) for t in _tensors(x[k])]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _tensors(v)]
    return []


def _clone(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


def _max_err(a, b):
    ta, tb = _tensors(a), _tensors(b)
    if len(ta) != len(tb) or not ta:
        return math.inf
    err = 0.0
    for x, y in zip(ta, tb):
        if x.shape != y.shape:
            return math.inf
        if x.numel():
            err = max(err, (x.detach().double() - y.detach().double()).abs().max().item())
    return err


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _rank_main(rank, a):
    import torch
    import torch.distributed as dist
    world = a["world_size"]
    device = "cpu"
    if a["device"] == "cuda":
        ids = [s for s in os.environ.get("MP_HARNESS_DEVICES", "").split(",") if s]
        torch.cuda.set_device(int(ids[rank]) if ids else rank)
        device = f"cuda:{torch.cuda.current_device()}"
    dist.init_process_group(a["backend"], init_method=f"tcp://127.0.0.1:{a['port']}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=a["timeout_s"]))
    try:
        mod = _load(a["cases"])
        ctx = argparse.Namespace(rank=rank, world_size=world, device=device, group=dist.group.WORLD)
        tol = a["tol"] if a["tol"] is not None else float(getattr(mod, "TOL", 1e-3))
        rows = []
        for case in _cases(mod):
            inputs = mod.make_inputs(case, ctx)
            ref = mod.seam("baseline", case, _clone(inputs), ctx)
            out = mod.seam("candidate", case, _clone(inputs), ctx)
            errs = [None] * world
            dist.all_gather_object(errs, _max_err(out, ref))
            ms = {}
            for leg in ("baseline", "candidate"):
                for _ in range(a["warmup"]):
                    mod.seam(leg, case, _clone(inputs), ctx)
                fresh = [_clone(inputs) for _ in range(a["iters"])]   # in-place seams get untouched inputs
                _sync(device)
                dist.barrier()
                t0 = time.perf_counter()
                for x in fresh:
                    mod.seam(leg, case, x, ctx)
                _sync(device)
                mine = (time.perf_counter() - t0) * 1e3 / max(1, a["iters"])
                per = [None] * world
                dist.all_gather_object(per, mine)
                ms[leg] = max(per)
            rows.append({"sig": str(case.get("sig")), "weight": case.get("weight"), "per_rank_err": errs,
                         "max_err": max(errs), "baseline_ms": ms["baseline"], "candidate_ms": ms["candidate"]})
        if rank == 0:
            with open(a["result_tmp"], "w") as fh:
                json.dump({"tol": tol, "cases": rows}, fh)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cases, world_size, backend="gloo", device="cpu", tol=None, iters=10, warmup=2, timeout_s=300, out=None):
    """Run every rank, then return the result dict (also written to ``out``)."""
    import torch.multiprocessing as mp
    tmp = tempfile.NamedTemporaryFile(prefix="mp_harness_", suffix=".json", delete=False)
    tmp.close()
    a = {"cases": os.path.abspath(cases), "world_size": world_size, "backend": backend, "device": device,
         "tol": tol, "iters": iters, "warmup": warmup, "timeout_s": timeout_s, "port": _free_port(),
         "result_tmp": tmp.name}
    res = {"world_size": world_size, "backend": backend, "cases": [], "correctness": "fail", "error": ""}
    ctx = mp.start_processes(_rank_main, args=(a,), nprocs=world_size, join=False, start_method="spawn")
    deadline = time.monotonic() + timeout_s * 2 + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(f"ranks did not finish within {timeout_s * 2 + 60}s")
        with open(tmp.name) as fh:
            body = json.load(fh)
        res.update(body, error="")
        ok = all(c["max_err"] <= body["tol"] for c in body["cases"]) and bool(body["cases"])
        res["correctness"] = "pass" if ok else "fail"
    except Exception as e:  # a failed or hung rank is a FAIL with its reason, never a hang
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        res["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
    finally:
        os.unlink(tmp.name)
    sp = [c["baseline_ms"] / c["candidate_ms"] for c in res["cases"] if c.get("candidate_ms")]
    res["geomean_speedup"] = math.exp(sum(map(math.log, sp)) / len(sp)) if sp else 0.0
    w = [(c.get("weight"), c["baseline_ms"] / c["candidate_ms"]) for c in res["cases"] if c.get("candidate_ms")]
    if w and all(isinstance(x, (int, float)) and x > 0 for x, _ in w):
        res["weighted_speedup"] = sum(x for x, _ in w) / sum(x / s for x, s in w)
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
    return res


def report(res):
    """The unified protocol lines."""
    errs = [c["max_err"] for c in res["cases"]]
    per_rank = [max(c["per_rank_err"][r] for c in res["cases"]) for r in range(res["world_size"])] if errs else []
    lines = [f"CORRECTNESS: {res['correctness'].upper()} max_err={max(errs) if errs else 'nan'} "
             f"tol={res.get('tol', 'nan')} per_rank={per_rank}" + (f" error={res['error']}" if res["error"] else "")]
    lines += [f"case {c['sig']}: {c['baseline_ms']:.4f} ms / {c['candidate_ms']:.4f} ms / "
              f"{c['baseline_ms'] / c['candidate_ms']:.3f}x" for c in res["cases"] if c.get("candidate_ms")]
    lines.append(f"GEAK_GEOMEAN_SPEEDUP={res['geomean_speedup']:.4f}")
    if "weighted_speedup" in res:
        lines.append(f"GEAK_WEIGHTED_SPEEDUP={res['weighted_speedup']:.4f}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", required=True, help="the cases module (.py) of the task")
    p.add_argument("--world-size", type=int, required=True, help="ranks = meta.required_gpu_count")
    p.add_argument("--backend", default="gloo", help="gloo (CPU) or nccl (GPU; RCCL on ROCm)")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="cuda: rank r uses MP_HARNESS_DEVICES[r] (default: device r)")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout-s", type=int, default=300, help="process-group init and collective timeout")
    p.add_argument("--out", default=None, help="result JSON")
    a = p.parse_args(argv)
    res = run(a.cases, a.world_size, a.backend, a.device, a.tol, a.iters, a.warmup, a.timeout_s, a.out)
    print("\n".join(report(res)), flush=True)
    return 0 if res["correctness"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
