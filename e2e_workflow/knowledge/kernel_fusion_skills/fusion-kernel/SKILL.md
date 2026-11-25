---
name: fusion-kernel
description: Write the fused kernel for one fusion region of a live LLM serving stack, behind the region's Python seam, in a GEAK kernel lane, by the co-optimization method (co-optimization/fusion_kernel_prompt.md steps 0-7). Establish the parity floor t_floor = min(chain, members tuned alone) first, budget and analyse every edge, design the partition, restore per-stage parity, then add the levers only fusion makes expressible; done means t_fused <= t_floor on every shape, else route that shape back. Guard the result to its measured envelope and keep it graph-capture safe. Used when e2e_workflow.js (KernelFusion on) hands a kernel lane a FUSION REGION task; the analysis that picks regions is the fusion-analysis skill.
---

# Fused kernel for a fusion region

A kernel lane (`kernel_lane.js`) reads this skill when `e2e_workflow.js` hands it a FUSION REGION task:
in the Milestone loop, or in HeadKernel for a region on a head's own kernel. The goal is one
implementation of the region's seam that is correct at the seam boundary and faster than what the
server runs today, on the shapes the server runs.

The analysis that chose the region (fusion-analysis skill) already compared it with optimizing its
kernels one by one and found fusion worth more. This skill does not repeat that analysis; it delivers
the best implementation the region's unit test can measure.

## What you are given

- **The lane task text**: `FUSION REGION <id> (<level>)` and a JSON object with the region (one line),
  `kernels_in_order` (the device kernels it launches today), `phases`, `m_range`, `seam`
  (`target_callable`, `source_file`), `inputs_outputs`, `launches_before`, `launches_after`,
  `suggested_design` and `bitwise_exact_expected`, plus the deployment's graph requirement and any
  operator constraints (they bind you).
- **The task dir**, built under the fusion UT contract (`e2e_workflow/roles/_fragments/fusion_ut_contract.md`):
  - `kernel_src/` holds the editable copy of the seam callable and the kernels it launches. It is the
    only thing you change.
  - The cases are the served shapes of every phase that calls the seam, each with its served weight.
  - The immutable unittest compares only the region's outputs at the seam boundary with the frozen live
    baseline, and times one call of the whole seam per case.

## Procedure

Read first, in this order: `../co-optimization/fusion_kernel_prompt.md` (the task and steps 0-7 below
in full, with the hardware and measurement facts), `../co-optimization/SKILL.md` §0.1 (what all six
measured fusions agree on) and §0.2 (which parts apply to this fusion type), then the references the
steps name (`02` levers, `04` rejected approaches and traps, `09` mid-fusion diagnosis). The steps are
the prompt's; this section maps them onto the GEAK task and its hand-back. Every step leaves a file.

**Done means `t_fused <= t_floor` on every case**, where `t_floor[case] = min(the chain as served,
the members each optimized alone)`. A fused kernel that is correct but slower than the floor is not done.
**"It saves a launch" is not a reason to fuse**: inside a captured graph one launch is ~2 µs, and the
gain comes from kernel-level work inside the fused structure (§0.1 items 2, 6).

0. **Baseline and parity floor** -> `kernel_src/_handback/baseline.json`.
   - Run the unittest on the unchanged `kernel_src/`: that is the chain as served (the baseline leg is
     the production path, graph-captured). Record per-case times; find which member takes most of the time.
   - Without single-member results in the task text, tune each member alone first (about an hour): Tier A
     another backend, Tier B explicit parameters bucketed by shape. Member tuning lives in `kernel_src/`:
     call the faster implementation explicitly (an aiter/CK kernel instead of a generic `torch.mm`, a
     hipBLASLt solution, a Triton/FlyDSL kernel for the served shapes) or pass the tile sizes, split-K,
     `num_warps`/`num_stages`/`waves_per_eu` keyed on the case's shape (a small table from M range to
     config); no serve-time autotuning. Server flags, environment variables and framework-loaded tuned-config
     tables belong to the run's config and tuning tracks, not here: they change every caller.
   - `t_floor[case] = min(chain, tuned alone)`. Record the structural findings of the tuning ("fewer splits
     win", "this algorithm is faster", "one launch is enough"): step 6 starts from them.
   - The tuned members alone are a deliverable in their own right (variant "tuned member"); a fused kernel
     that only beats the stock chain hides a tuning gain behind a fusion.
1. **Budget and floors** -> `design.md` §1. Fusion upper bound ≈ (graph nodes removed) × ~2 µs +
   intermediate bytes ÷ 7 TB/s. Per stage: required bytes, FLOP and serial hops -> a floor (`09` §3);
   a stage at ≥ 0.85 of its floor only gets faster by moving fewer bytes. If the bound cannot pay for the
   expected costs (sync, recompute, lost parallelism), say so and plan the per-shape fallback or no fusion.
2. **Every edge** -> `design.md` §2: the `06` edge template plus a dependency class D0 local / D1 recomputable
   / D2 small-group reduction / D3 global reduction / D4 broadcast-reuse, and the three conditions of `01`
   §2. Name the kernel-level reason to fuse (prefetch under the producer, on-chip or same-XCD handover,
   tail filling, claim order for reuse). **D3/D4 edges must try the rewrites first** (another algorithm,
   fewer splits on one XCD, last-arriver does the consumer's work, norm ownership N1, algebraic split A10,
   moving the boundary, e.g. quant into the producer's epilogue N4), each recorded as tried; when all fail
   the verdict for that edge is "no fusion, route per shape".
3. **Design space** -> `design.md` §3: at least 4 partitions (the producer's, the consumer's, recompute
   instead of sync, small split + same-XCD handover or persistent/last-arriver, per-shape routing), each with
   CTAs and waves against 256 CUs / 8 XCDs, sync type and scope, recompute factor, VGPR/AGPR/LDS at 1
   wave/SIMD, a time estimate. Pick a primary and a fallback.
4. **Skeleton**: reuse each member's tuned body, change only ownership, placement, queueing and publishing;
   every constant through a shape-bucket selector; the partition itself a parameter. Sync by readiness
   counters / last-arriver, never grid barriers (`02` C1, C9). Read the metadata and ISA at once (`09` §5).
5. **Parity before levers**: time every stage inside the fused kernel (`tools/region_stamp.h` +
   `region_timeline.py` for HIP, or floor probes). Any stage slower than it runs tuned alone is a defect;
   fix it before step 6 (§0.1 item 6; `09` §4 symptoms S1–S13).
6. **Fusion levers and re-tuning, in rounds** (`09` §1): timeline -> floors -> the stage with the largest
   gap on the critical path -> its symptom -> one lever behind a compile-time switch -> paired A/B -> record,
   including what not to retry. Re-tune constants after every structural lever. Do not retry what `04` already
   lost without a new reason (extra early reads, cross-XCD stealing, grid barriers, "one launch" as a goal,
   rewriting a tuned stage in another DSL). Stop, with evidence, when every case meets `t_floor`, or 2–3
   rounds gain under ~2–3%, or a floor probe shows the bottom, or every D3/D4 rewrite failed.
7. **Accept and attribute** -> `ut_result.json`, `notes.md`, `support.json` under `kernel_src/_handback/`:
   - correctness on every case (the region UT; bit-exact where `meta.acceptance.exact_outputs` says, and the
     same input twice gives the same bits); graph capture and replay work;
   - per case the **five columns** in `ut_result.json` `shapes`: chain, tuned alone, floor, first fused,
     final fused (µs), plus `correctness`, `weighted_speedup`, the routing table and VGPR/LDS/occupancy;
   - a case where the fused kernel loses to the floor is routed to the floor implementation inside
     `kernel_src/` and listed in `support.json` `refusals`;
   - `notes.md` states **one of three verdicts**: all cases meet the floor / some do and the rest route back /
     fusion not feasible (with the stop-condition evidence; then deliver the floor implementation, never a
     slower fused one), and the fusion-off / fusion-on attribution.

**Guard and graph safety** (every step): call the new path only inside the envelope the unittest measured it
correct and faster in (phase, rows / M range, dtypes, shapes); everything else goes to the stock path. When
the server captures decode into a CUDA/HIP graph: launch on the current stream, no host sync, no `.item()`,
no Python branch on a GPU value, no allocation inside the captured region; the guard branches on host-side
values only. `init.py` (a `prepare(ctx)`) only when start-up preparation is needed (JIT of every width
before the KV cache is sized, registrations).

## The hand-back

Every implementer of a fusion region, a kernel lane or an outside team, hands back the same directory:
`kernel_src/`, `support.json`, `init.py` (optional), `ut_result.json`, `notes.md`. For a kernel lane the
workflow assembles it (`e2e_workflow/scripts/fusion_handback.py assemble`): `kernel_src/` is the task's
`kernel_src/` with your final patch applied, and every file you wrote under `kernel_src/_handback/`
(`support.json`, `notes.md`, `init.py`, `ut_result.json`, `baseline.json`, `design.md`) moves to the top of it;
without your `ut_result.json`, the lane's own validation of the same unittest stands in (no five columns). It is checked against the region UT
(layout, `support.json`, the code compiles, `meta.acceptance.min_speedup`) before the e2e A/B. Without your
`support.json` the workflow derives one from the region's phases and rows and marks it `derived`: state the
real envelope instead.

## Rules

- The outputs at the seam boundary must match the frozen baseline within the task's tolerance, and
  bit-exactly when the task says so. Intermediates between member kernels are not checked.
- Change only `kernel_src/`. The seam's signature, its return values and any declared in-place
  mutation stay as they are.
- A member tuned through a server flag, an environment variable or a tuned-config table outside
  `kernel_src/` is not delivered here: the run's config and tuning tracks own those. Inside this task,
  member tuning means the code in `kernel_src/` calls a faster implementation or passes explicit
  parameters (step 0).
- Fusion removes waiting, never work: fewer HBM round trips, on-chip handover and overlap are the win, a
  removed launch alone is ~2 µs in a graph. Do not trade them for a loss in parallelism the served shapes
  need (split-K, split-KV) without measuring it.

## Choosing a language

| Use | When |
|---|---|
| Triton | The default for elementwise, reduction and epilogue fusions that Triton expresses directly. |
| FlyDSL | The fusion needs control Triton does not expose (an exact MFMA atom, an LDS layout, cross-lane shuffles, buffer loads with OOB checks), or a FlyDSL reference kernel for the same op family exists. See [FlyDSL](references/flydsl/README.md). |
| The library's own fused variant | The library (aiter, CK, hipBLASLt) already ships a kernel or an epilogue option that covers the region. Calling it is a valid fused implementation, judged against the floor like any other. |

The task text's language is a starting point, not a mandate; the unittest decides.

## References

- [co-optimization](../co-optimization/SKILL.md) and its [fusion kernel prompt](../co-optimization/fusion_kernel_prompt.md):
  the method above, with references `01`–`09` and `tools/`.
- [FlyDSL](references/flydsl/README.md): FlyDSL authoring, with [API](references/flydsl/api.md),
  [fusion recipes](references/flydsl/fusion-recipes.md), [tuning](references/flydsl/tuning.md) and
  [debugging](references/flydsl/debugging.md).
