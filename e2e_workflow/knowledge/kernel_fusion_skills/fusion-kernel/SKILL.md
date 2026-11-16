---
name: fusion-kernel
description: Write the fused kernel for one fusion region of a live LLM serving stack, behind the region's Python seam, in a GEAK kernel lane. Tune the region's heavy member first, fuse its neighbours on top of it, keep the best of stock / tuned member / tuned and fused at the region unit test, guard the result to its measured envelope and keep it graph-capture safe. Used when e2e_workflow.js (KernelFusion on) hands a kernel lane a FUSION REGION task; the analysis that picks regions is the fusion-analysis skill.
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

1. **Baseline.** Read the region, `meta.json` and the cases. Run the unittest on the unchanged
   `kernel_src/` and record the per-case times. Find which member kernel takes most of the region's time.
2. **Tune the heavy member first** when one member dominates the region (a GEMM, attention, MoE or other
   library op, typically at L2). Make it as fast as you can on its own, inside `kernel_src/`, with the
   rest of the region unchanged. This is variant B, "tuned member". See
   [tune-then-fuse](references/tune-then-fuse.md).
3. **Fuse on top of the best member.** Fold the neighbours (bias, activation, scale, quant, norm, residual,
   RoPE, cache write) into the prologue or epilogue of the best member implementation from step 2. Keep
   intermediates in registers or LDS; the fused kernel reads the region's inputs once and writes its
   outputs once. This is variant C, "tuned and fused". Patterns: [fusion recipes](references/flydsl/fusion-recipes.md).
4. **Pick the best variant at the unit test.** Compare A (stock), B (tuned member) and C (tuned and
   fused) by the unittest's weighted speedup, after correctness. C must beat B, not only A: a fusion
   that only wins against the stock member hides a tuning gain behind a fusion. Deliver the best
   correct variant in `kernel_src/`, even when it is B.
5. **Guard it.** Inside the seam implementation in `kernel_src/`, call the new path only inside the
   envelope the unittest measured it correct and faster in (phase, row count / M range, dtypes, shapes).
   Every other call goes to the stock path: another phase, a larger M such as a prefill chunk, a batch
   size you did not time, inputs it does not support.
6. **Keep it graph-capture safe** when the server captures decode into a CUDA/HIP graph: launch on the
   current stream, no host sync, no `.item()`, no Python branch on a GPU value, no allocation inside the
   captured region. The guard in step 5 must branch on host-side values only (shapes, flags).
7. **Report** in your notes: the variant you delivered (A, B or C) and the weighted speedup of each one
   you measured; per-case times; the envelope; launches before and after; whether the outputs are
   bit-exact; served shapes or phases the cases lack, if you noticed any (the unittest is immutable: guard
   off them instead of adding cases).

## Rules

- The outputs at the seam boundary must match the frozen baseline within the task's tolerance, and
  bit-exactly when the task says so. Intermediates between member kernels are not checked.
- Change only `kernel_src/`. The seam's signature, its return values and any declared in-place
  mutation stay as they are.
- A member tuned through a server flag, an environment variable or a tuned-config table outside
  `kernel_src/` is not delivered here: the run's config and tuning tracks own those. Inside this task,
  member tuning means the code in `kernel_src/` calls a faster implementation or passes explicit
  parameters (see [tune-then-fuse](references/tune-then-fuse.md)).
- Fewer launches and fewer HBM round trips are the win. Do not trade them for a loss in parallelism the
  served shapes need (split-K, split-KV) without measuring it.

## Choosing a language

| Use | When |
|---|---|
| Triton | The default for elementwise, reduction and epilogue fusions that Triton expresses directly. |
| FlyDSL | The fusion needs control Triton does not expose (an exact MFMA atom, an LDS layout, cross-lane shuffles, buffer loads with OOB checks), or a FlyDSL reference kernel for the same op family exists. See [FlyDSL](references/flydsl/README.md). |
| The library's own fused variant | The library (aiter, CK, hipBLASLt) already ships a kernel or an epilogue option that covers the region. Calling it is a valid variant C. |

The task text's language is a starting point, not a mandate; the unittest decides.

## References

- [Tune, then fuse](references/tune-then-fuse.md): tuning the heavy member inside the region task, and the three-variant comparison.
- [FlyDSL](references/flydsl/README.md): FlyDSL authoring, with [API](references/flydsl/api.md),
  [fusion recipes](references/flydsl/fusion-recipes.md), [tuning](references/flydsl/tuning.md) and
  [debugging](references/flydsl/debugging.md).
