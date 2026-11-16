# Tune, then fuse

A region with one heavy member (a GEMM plus its quant, attention plus its pre-processing, a MoE GEMM plus
its activation) has two sources of gain: a faster member, and fewer launches and HBM round trips around
it. Fusing first mixes the two, and a fused kernel built on the stock member can lose to the tuned member
alone. So the member is tuned first, and the fusion is judged against the tuned member.

## The three variants

| Variant | `kernel_src/` contains | Judged against |
|---|---|---|
| A, stock | The extracted seam, unchanged | — (the unittest's baseline) |
| B, tuned member | The seam, calling a faster implementation of the heavy member; the other members unchanged | A |
| C, tuned and fused | One kernel (or fewer kernels) that does the tuned member's work and its neighbours' | B and A |

Deliver the best correct one. Report the weighted speedup of each variant you built.

## Tuning the member inside the task

Member tuning must live in `kernel_src/`, because the region's seam is what the server rebinds:

- **Another backend for the member.** Call the faster implementation explicitly: an aiter or CK kernel
  in place of a generic `torch.mm`, a hipBLASLt solution, a Triton or FlyDSL kernel written for the
  served shapes.
- **Explicit parameters.** Pass the tile sizes, split-K, `num_warps` / `num_stages`, `waves_per_eu` or
  the solution index the served shapes want, keyed on the shape inside `kernel_src/` (a small table from
  M range to config). Do not rely on autotuning at serve time.
- **Shapes.** Tune on the unittest's cases, which carry the served M range per phase. A decode or verify
  M (a few to a few hundred rows) usually wants a different config from a prefill chunk.

Not here: server flags, environment variables and tuned-config tables loaded by the framework. Those
belong to the run's config and tuning tracks, and a change to them would affect every caller, not only
this seam.

## Fusing on the tuned member

- Keep the tuned member's tile configuration; add the neighbours in its prologue (input norm, dequant)
  or epilogue (bias, activation, scale, quant, residual, split-K combine).
- An epilogue that needs a full-row statistic (a norm after a GEMM) usually forces a different tiling.
  Measure it; when it loses, keep B and fuse only what fits the tuned tile.
- Re-check correctness after every change, then time with graph replay on every case.

## Example: GEMM + quant (L2)

1. A: the stock GEMM, then a separate per-token or per-group quant kernel.
2. B: the GEMM called with a backend or config tuned for the served M range; the quant kernel unchanged.
3. C: the tuned GEMM with the quant in its epilogue (scale and convert in f32 registers, clamp before the
   fp8 conversion), writing the quantized output and its scales once.
4. Deliver C only when it beats B on the weighted cases; otherwise deliver B.
