"""mp_harness.py: the multi-rank harness of a fusion region UT, on 2 CPU ranks over gloo (no GPU)."""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
    pytest.skip("torch.distributed with gloo is required", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mp_harness  # noqa: E402

# A toy seam: the all-reduce of each rank's partial sum (what a TP row-parallel layer ends with). The baseline is
# torch's all_reduce; the candidate re-implements it with all_gather + a local sum. BREAK names a negative control.
CASES = '''
import torch
import torch.distributed as dist
CASES = [{"sig": "decode_bs4", "weight": 3}, {"sig": "prefill_256", "weight": 1}]
TOL = 1e-6
BREAK = %r

def make_inputs(case, ctx):
    n = 4 if case["sig"].startswith("decode") else 256
    g = torch.Generator().manual_seed(1000 + ctx.rank)          # distinct per rank
    return {"x": torch.randn(n, 64, generator=g, dtype=torch.float64)}

def seam(leg, case, inputs, ctx):
    x = inputs["x"]
    if leg == "baseline":
        dist.all_reduce(x)
        return x
    if BREAK == "missing_comm":                                  # never talks to its peers
        return x * ctx.world_size
    parts = [torch.empty_like(x) for _ in range(ctx.world_size)]
    dist.all_gather(parts, x)
    y = torch.stack(parts).sum(0)
    if BREAK == "single_rank" and ctx.rank == 1:                  # one rank off by 4*tol
        y = y + 4 * TOL * y.abs().max()
    return y
'''


def _cases(tmp_path, brk=None):
    p = tmp_path / f"cases_{brk or 'ok'}.py"
    p.write_text(CASES % brk)
    return str(p)


def test_the_unchanged_seam_passes_on_every_rank_and_reports_the_protocol(tmp_path):
    out = tmp_path / "result.json"
    res = mp_harness.run(_cases(tmp_path), 2, iters=3, warmup=1, timeout_s=60, out=str(out))
    assert res["correctness"] == "pass" and not res["error"] and out.is_file()
    assert [c["sig"] for c in res["cases"]] == ["decode_bs4", "prefill_256"]
    assert all(len(c["per_rank_err"]) == 2 and c["max_err"] <= 1e-6 for c in res["cases"])
    assert res["geomean_speedup"] > 0 and res["weighted_speedup"] > 0
    lines = mp_harness.report(res)
    assert lines[0].startswith("CORRECTNESS: PASS") and "per_rank=[" in lines[0]
    assert lines[1].startswith("case decode_bs4: ") and lines[1].endswith("x") and " ms / " in lines[1]
    assert any(l.startswith("GEAK_GEOMEAN_SPEEDUP=") for l in lines)
    assert any(l.startswith("GEAK_WEIGHTED_SPEEDUP=") for l in lines)


@pytest.mark.parametrize("brk", ["missing_comm", "single_rank"])
def test_a_negative_control_fails(tmp_path, brk):
    res = mp_harness.run(_cases(tmp_path, brk), 2, iters=1, warmup=0, timeout_s=60)
    assert res["correctness"] == "fail" and not res["error"]
    if brk == "single_rank":   # only rank 1 deviates: the per-rank errors show which, the max over ranks fails it
        assert all(c["per_rank_err"][0] <= 1e-6 < c["per_rank_err"][1] for c in res["cases"])


def test_the_cli_exits_1_on_a_failed_control_and_0_on_a_pass(tmp_path):
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "mp_harness.py"), "--world-size", "2",
           "--iters", "1", "--warmup", "0", "--timeout-s", "60"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    ok = subprocess.run(cmd + ["--cases", _cases(tmp_path)], capture_output=True, text=True, timeout=300, env=env)
    bad = subprocess.run(cmd + ["--cases", _cases(tmp_path, "missing_comm")], capture_output=True, text=True,
                         timeout=300, env=env)
    assert ok.returncode == 0 and "CORRECTNESS: PASS" in ok.stdout and "GEAK_GEOMEAN_SPEEDUP=" in ok.stdout, ok.stderr
    assert bad.returncode == 1 and "CORRECTNESS: FAIL" in bad.stdout


def test_a_rank_that_raises_is_a_fail_with_its_reason_not_a_hang(tmp_path):
    p = tmp_path / "cases_raise.py"
    p.write_text(CASES % None + "\ndef make_inputs(case, ctx):\n    raise RuntimeError('no weights for rank')\n")
    res = mp_harness.run(str(p), 2, iters=1, warmup=0, timeout_s=30)
    assert res["correctness"] == "fail" and "no weights for rank" in res["error"]
