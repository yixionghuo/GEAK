"""fusion_handback.py: one hand-back layout for every implementer of a fusion region (no GPU)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fusion_handback as fh  # noqa: E402

SUPPORT = {"phases": ["decode", "target_verify"], "m_range": "1-256", "dtypes": ["bf16"], "refusals": ["prefill"]}
UT_OK = "import sys\nprint('CORRECTNESS: PASS')\nprint('GEAK_WEIGHTED_SPEEDUP=1.25')\nsys.exit(0)\n"


def _task(root, meta_extra=None, unittest_src=UT_OK):
    t = os.path.join(root, "task")
    os.makedirs(os.path.join(t, "kernel_src"))
    with open(os.path.join(t, "kernel_src", "fused.py"), "w") as f:
        f.write("def run(x):\n    return x\n")
    meta = {"fusion_contract": True, "level": "L1",
            "candidate_bind": {"kind": "rebind", "target": "pkg.m:f", "file": "kernel_src/fused.py"}}
    meta.update(meta_extra or {})
    with open(os.path.join(t, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(t, "unittest.py"), "w") as f:
        f.write(unittest_src)
    os.makedirs(os.path.join(t, "controls", "perturb_out"))
    with open(os.path.join(t, "controls", "controls.json"), "w") as f:
        json.dump({"perturb_out": {"must_fail": ["eager"]}}, f)
    return t


def _lane_patch(root, task):
    """A lane's final_patch.diff: the new kernel plus the hand-back files under kernel_src/_handback/."""
    ws = os.path.join(root, "ws")
    subprocess.run(["cp", "-a", task, ws], check=True)
    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(g + ["init", "-q"], cwd=ws, check=True)
    subprocess.run(g + ["add", "-A"], cwd=ws, check=True)
    subprocess.run(g + ["commit", "-qm", "base"], cwd=ws, check=True)
    with open(os.path.join(ws, "kernel_src", "fused.py"), "w") as f:
        f.write("def run(x):\n    return x * 1  # fused\n")
    os.makedirs(os.path.join(ws, "kernel_src", "_handback"))
    with open(os.path.join(ws, "kernel_src", "_handback", "support.json"), "w") as f:
        json.dump(SUPPORT, f)
    with open(os.path.join(ws, "kernel_src", "_handback", "notes.md"), "w") as f:
        f.write("Delivered C (tuned and fused).\n")
    subprocess.run(g + ["add", "-A"], cwd=ws, check=True)
    patch = os.path.join(root, "final_patch.diff")
    with open(patch, "w") as f:
        subprocess.run(["git", "--no-pager", "diff", "--cached", "HEAD"], cwd=ws, check=True, stdout=f)
    return patch


class HandbackTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.task = _task(self.root)
        self.out = os.path.join(self.root, "handback")

    def test_assemble_from_a_lane_then_check(self):
        patch = _lane_patch(self.root, self.task)
        res = fh.assemble(self.task, patch, {"validation_status": "accepted", "final_weighted": 1.3}, self.out)
        self.assertTrue(res["ok"], res)
        self.assertIn("# fused", open(os.path.join(self.out, "kernel_src", "fused.py")).read())
        self.assertFalse(os.path.exists(os.path.join(self.out, "kernel_src", "_handback")))
        self.assertEqual(json.load(open(os.path.join(self.out, "support.json"))), SUPPORT)
        self.assertEqual(json.load(open(os.path.join(self.out, "ut_result.json")))["source"], "kernel_lane")
        chk = fh.check(self.task, self.out)
        self.assertTrue(chk["ok"], chk)
        self.assertAlmostEqual(chk["speedup"], 1.3)
        self.assertEqual(chk["warnings"], [])

    def test_the_task_dir_is_never_touched(self):
        patch = _lane_patch(self.root, self.task)
        fh.assemble(self.task, patch, {"validation_status": "accepted", "final_weighted": 1.3}, self.out)
        self.assertEqual(open(os.path.join(self.task, "kernel_src", "fused.py")).read(), "def run(x):\n    return x\n")

    def test_without_support_json_rejected_unless_the_region_fills_it_in(self):
        res = fh.assemble(self.task, "", {"validation_status": "accepted", "final_weighted": 1.3}, self.out)
        self.assertTrue(res["ok"])
        chk = fh.check(self.task, self.out)
        self.assertFalse(chk["ok"])
        self.assertTrue(any("support.json" in p for p in chk["problems"]))
        out2 = os.path.join(self.root, "handback2")
        res = fh.assemble(self.task, "", {"validation_status": "accepted", "final_weighted": 1.3}, out2,
                          {"phases": ["decode"], "m_range": "1-64"})
        self.assertTrue(any("derived" in w for w in res["warnings"]))
        sup = json.load(open(os.path.join(out2, "support.json")))
        self.assertEqual((sup["phases"], sup["m_range"], sup["derived"]), (["decode"], "1-64", True))
        self.assertTrue(fh.check(self.task, out2)["ok"])

    def _outside(self, support=SUPPORT, ut=None, src="def run(x):\n    return x\n"):
        os.makedirs(os.path.join(self.out, "kernel_src"))
        with open(os.path.join(self.out, "kernel_src", "fused.py"), "w") as f:
            f.write(src)
        with open(os.path.join(self.out, "support.json"), "w") as f:
            json.dump(support, f)
        with open(os.path.join(self.out, "notes.md"), "w") as f:
            f.write("outside team\n")
        with open(os.path.join(self.out, "ut_result.json"), "w") as f:
            json.dump(ut or {"correctness": "pass", "speedup": 1.25}, f)

    def test_an_outside_hand_back_is_checked_the_same_way_and_rerun(self):
        self._outside()
        chk = fh.check(self.task, self.out, run=True)
        self.assertTrue(chk["ok"], chk)
        self.assertEqual(chk["rerun"]["correctness"], "pass")
        self.assertAlmostEqual(chk["rerun"]["speedup"], 1.25)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "ut_result.kf.json")))

    def test_a_failing_rerun_is_rejected(self):
        self.task = _task(os.path.join(self.root, "b"), unittest_src="import sys\nsys.exit(1)\n")
        self._outside()
        chk = fh.check(self.task, self.out, run=True)
        self.assertFalse(chk["ok"])

    def test_code_that_does_not_compile_or_misses_the_bind_file(self):
        self._outside(src="def run(x)\n")
        chk = fh.check(self.task, self.out)
        self.assertTrue(any("does not compile" in p for p in chk["problems"]))
        os.remove(os.path.join(self.out, "kernel_src", "fused.py"))
        with open(os.path.join(self.out, "kernel_src", "other.py"), "w") as f:
            f.write("x = 1\n")
        chk = fh.check(self.task, self.out)
        self.assertTrue(any("candidate_bind.file" in p for p in chk["problems"]))

    def test_min_speedup_and_a_failed_ut(self):
        self.task = _task(os.path.join(self.root, "c"), meta_extra={"acceptance": {"min_speedup": 1.1}})
        self._outside(ut={"correctness": "pass", "speedup": 1.05})
        self.assertTrue(any("min_speedup" in p for p in fh.check(self.task, self.out)["problems"]))
        with open(os.path.join(self.out, "ut_result.json"), "w") as f:
            json.dump({"correctness": "fail", "speedup": 1.5}, f)
        self.assertTrue(any("passing UT" in p for p in fh.check(self.task, self.out)["problems"]))

    def test_support_json_fields(self):
        self._outside(support={"phases": [], "refusals": "prefill"})
        probs = " ".join(fh.check(self.task, self.out)["problems"])
        self.assertIn("phases", probs)
        self.assertIn("refusals", probs)
        self.assertIn("m_range", probs)

    def test_a_multi_rank_hand_back_states_the_ranks_its_ut_runs(self):
        self.task = _task(os.path.join(self.root, "m"), meta_extra={"level": "L3", "mega": True, "required_gpu_count": 4})
        self._outside()
        self.assertTrue(any("required_gpu_count must be 4" in p for p in fh.check(self.task, self.out)["problems"]))
        with open(os.path.join(self.out, "support.json"), "w") as f:
            json.dump({**SUPPORT, "required_gpu_count": 4}, f)
        self.assertTrue(fh.check(self.task, self.out)["ok"])
        out2 = os.path.join(self.root, "handback_m")
        fh.assemble(self.task, "", {"validation_status": "accepted", "final_weighted": 1.3}, out2,
                    {"phases": ["decode"], "m_range": "1-64", "required_gpu_count": 4})
        self.assertEqual(json.load(open(os.path.join(out2, "support.json")))["required_gpu_count"], 4)
        self.assertTrue(fh.check(self.task, out2)["ok"])

    def test_cli_prints_one_json_object(self):
        patch = _lane_patch(self.root, self.task)
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "fusion_handback.py"), "assemble",
                            "--task-dir", self.task, "--final-patch", patch, "--out", self.out, "--check",
                            "--lane-result", json.dumps({"validation_status": "accepted", "final_weighted": 1.3})],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(json.loads(p.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
