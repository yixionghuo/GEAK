# Fusion UT contract

Read by the UT agent of a fusion region, a kernel candidate of `e2e_workflow.js` that names a region of
the run's fusion profile (KernelFusion switched on): the kernel-extract skill on an AgentX trace replay
(`kernel_extractor`, PHASE=prepare), `kernel_extractor` PHASE=extract otherwise (in the Milestone loop, or in
HeadKernel for a region on a head's own kernel).
Your task exists so that the next step can REPLACE the whole region with fewer (usually one) kernel
launches: the GEAK kernel layer (`kernel_lane.js`) optimizes the task you build, and the e2e integrator
rebinds the region's seam (`candidate_bind`) to the result on the live server.

## FUSION UT CONTRACT

The UT must not pin the current kernel structure:

1. `kernel_src/` holds an editable copy of the seam callable (the host-side function whose body launches
   the region) together with the kernels it launches; the case runner and the timer call that seam
   callable from kernel_src with the recorded inputs -- never the member kernels directly -- so a
   candidate may change how many kernels the region launches, their grid/split and their intermediates.
2. Correctness compares only the region's outputs at the seam boundary (and any declared in-place
   mutation) with the frozen live baseline; intermediates between member kernels are not checked.
3. Execution/binding proof that every member kernel fires is required for the BASELINE leg only; for the
   candidate leg require only that the kernel_src seam implementation ran, never that the original member
   kernels still launch or keep their count/split.
4. The negative control mutates the OUTPUT of the kernel_src seam implementation (e.g. perturb what the
   host function returns or writes), so it still triggers after the members are fused away.
5. Time one call of the whole seam per case (launch overhead included) -- the win of a fusion is fewer
   launches and fewer HBM round-trips.
6. The cases cover EVERY phase in which the live server calls the seam (prefill / extend chunks of every
   served width, decode, target verify, draft), each with its served shapes and weight -- not only the
   phase the region was found in: a fused kernel that wins at verify rows can lose badly at prefill
   widths. Find them from the trace digest's step kinds and the source, not only from the region's listed
   phases, and check the cases against the served distribution (`WORKLOAD_SUMMARY` /
   `PROFILE_WORKLOAD_JSON` when given, else the baseline server log, and the region's phases / m_range):
   add every served shape with real weight that the cases lack, and weight each case by its served share.

Record `"fusion_contract": true` in meta.json.

## The unified task dir (L1–L4 share it)

Every fusion region's UT, whatever its level and whoever implements it, is one GEAK task dir with these
additions. L1–L3 use what is below; L3 MegaKernel and L4 add their layer-level fields later.

- **`meta.json`**, beside the fields GEAK already writes:
  - `fusion_contract: true`, `level` (`L1` / `L2` / `L3`);
  - `analysis`: `"analysis.json"`;
  - `cases`: one entry per served case, `{"sig", "step_type", "width", "context", "weight"}` (step_type:
    prefill_chunk / decode / target_verify / draft; weight = its served share), the same signatures the
    timing cases use;
  - `regime.speculative`: `{"algorithm", "num_draft_tokens"}` from the server flags, `null` without
    speculative decoding;
  - `acceptance`: `{"min_speedup", "exact_outputs", "noise_bound"}`: the lowest weighted speedup a
    hand-back must reach at this UT (1.0 unless `OPERATOR_CONSTRAINTS` says otherwise: the e2e A/B
    decides the rest), the outputs that must match bit for bit (in-place writes such as a KV-cache slot;
    `[]` when none), and the measured noise of the baseline against itself (`run/self_noise.json`).
- **`analysis.json`**: the region's entry of the fusion analysis, as given in `FUSION_ANALYSIS` (or the
  `analysis` of the `FUSION_SCOPE` entry): why it is fused, at what level, the expected gains of both
  routes. Copy it; do not re-derive it.
- **`controls/`**: the negative controls as files another implementer can re-run. Each control is a
  directory `controls/<name>/` holding the files that replace their namesakes in `kernel_src/` (the seam
  implementation with one thing broken); `controls/controls.json` lists them,
  `{"<name>": {"what": "...", "must_fail": ["eager", ...]}}`. At least `perturb_output`: the seam's output
  scaled by `1 + 4·tol`. Run every control with the task's own `unittest.py` (a copy of the task dir whose
  `kernel_src/` files are replaced by the control's): each must exit 1 (correctness FAIL; a build or
  import error does not count). Record the runs in `run/controls_result.json`, and point the source-delivery
  record's `negative_control_ref` at it.
- No `selfcheck.json`: the self-check is `run/ut_result.json` (the unchanged `kernel_src/` passes with
  speedup ≈ 1.0), `run/self_noise.json` and `run/controls_result.json`.

## The hand-back the UT is accepted against

Whoever implements the region hands back one directory (`scripts/fusion_handback.py`): `kernel_src/` (the
only change), `support.json` (`phases`, `m_range` or `widths`, `dtypes`, `refusals`: where it engages and
where it falls back to the stock path), `init.py` (optional), `ut_result.json` (this UT's result on it) and
`notes.md`. A GEAK kernel lane produces it through the fusion-kernel skill. KernelFusion checks it against
this task dir (`acceptance.min_speedup`, the candidate_bind file) before the e2e A/B.

## AgentX: the kernel-extract skill (PHASE=prepare)

Explicit user scope (overrides the default selection): build exactly ONE task for the single entry of
FUSION_SCOPE, named by its `task`. The entry is a FUSION REGION: declare its device_kernels as one joint
boundary (joint_ut.phases) owned by that task, reached through the entry's seam (or the narrowest
enclosing live callable; record why). The candidate leg will rebind that callable to one fused
implementation, so the oracle is taken at the region boundary. Reuse this run's evidence
(PREVIOUS_WORKLOAD_JSON / PREVIOUS_RANKING_JSON / PREVIOUS_UT_DELIVERY and the traces under TRACE_DIR);
do not launch a new full replay. This region is your ONLY job and the time in REMAINING_MIN is all
yours: do not defer or stop early for budget. When an input is hard to reconstruct (quantized checkpoint
operands, split/partial buffers, library-op wrappers), build it the way the live path does -- load the
real tensors from MODEL_PATH, call the live producer, or bind below the wrapper -- and verify by running
it. Other regions are prepared in parallel on the other GPUs: use only GPU_ID for UT work and write only
under EVIDENCE_DIR and your task dir. Complete source extraction, UT authoring and GPU validation, keep a
failure in the delivery with its concrete blocker, pack with `agentx_kernel_preparation.mjs
--emit-prepared` and return ONLY {delivery_path, prepared} (prepared = that command's stdout VERBATIM).
Do not optimize or integrate.

## Other workloads: kernel_extractor (PHASE=extract)

Build the immutable unittest task dir for a FUSION REGION, not a single kernel. The baseline leg is the
live, unfused region reached through ONE Python seam (KERNEL.target_callable, a function whose body
launches every kernel in KERNEL.device_kernels); candidate_bind rebinds that seam, so the candidate leg
runs the fused implementation from kernel_src/. Record I/O at the seam boundary (KERNEL.inputs_outputs)
for the oracle. If the given seam does not cover the region, use the narrowest enclosing callable that
does and say so in notes; if none exists return editable:false with notes "no_rebind_seam".
