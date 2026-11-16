# FlyDSL kernel authoring, fusion and optimization

Reference for the fusion-kernel skill (`../../SKILL.md`) when a fused kernel, or a member kernel tuned
before fusing, is written in FlyDSL.

FlyDSL is a Python DSL and MLIR stack for GPU kernels with **explicit layouts and tiling**. A kernel is
written in Python. The Fly dialect lowers its layout algebra to arithmetic, then the pipeline goes through
ROCDL and LLVM to HSACO. It sits below Triton: the author controls the thread → value mapping, LDS,
copy and MMA atoms, and scheduling hints. Compilation is JIT and takes seconds, unlike CK's C++ templates.

Sources: the ROCm/FlyDSL repository (README, `docs/kernel_authoring_guide.md`, `docs/kernel_tuning_guide.md`,
`.claude/skills/*`, `kernels/*`) and the FlyDSL introduction deck (2026-04-10). The API evolves:
check every name against the installed `flydsl` before relying on it.

## 1. When to use FlyDSL

| Choose | When |
|---|---|
| **FlyDSL** | The target needs control Triton does not expose: an exact MFMA/WMMA atom, an LDS layout or swizzle, buffer loads with OOB checks, cross-lane shuffles, scheduling hints, a resident/persistent design. Also when a FlyDSL reference kernel for the same op family already exists (§4). |
| **Triton** | A block-level elementwise or reduction fusion that Triton expresses directly, and the measured Triton kernel is already close to the memory roofline. |
| **HIP / CK / aiter** | The production path already uses a tuned library kernel and the change is a configuration of it. |

The workflow decides the winner by measurement. This skill is a prior, not a mandate.

## 2. Requirements

1. **Correctness first.** A kernel is tuned only after it matches the reference on every shape the server
   runs, including edge widths and tails. The comparison is at the region boundary for a fusion, and at
   the op boundary for a single kernel.
2. **Deployed regime.** Shapes, dtypes (fp8 is `e4m3fnuz` on gfx942 and `e4m3fn` on gfx950), arch and
   CUDA-graph capture are as the server runs them.
3. **Graph-capture safe.** Launch on the current torch stream. No host sync, no `.item()`, no Python
   branch on a GPU value, no allocation inside the captured region.
4. **Measure with a compiled handle.** Every call of a `@flyc.jit` function retraces before the cache
   lookup, so benchmarks keep `compiled = flyc.compile(launch, *args)` and call `compiled(...)`.
5. **One variable per experiment.** Change one knob, re-verify correctness, re-measure. gfx950 benchmark
   noise is about ±14%: use 60–80 iterations and the median of 7 runs.

## 3. Procedure

1. **Identify the arch.** `gfx942` (MI300X, LDS 64 KB), `gfx950` (MI350/MI355X, LDS 160 KB), `gfx1250`
   (wave32, LDS 320 KB, TDM async copy), `gfx1201` (RDNA4, LDS 64 KB). gfx9xx waves are 64 lanes.
2. **Classify the kernel.** Elementwise, reduction (norm, softmax), tiled copy, GEMM, attention, or fused
   (a combination). The class picks the skeleton (`api.md` §8).
3. **Start from the nearest reference kernel** (§4). Do not write a GEMM, attention or MoE kernel from
   scratch when one exists. Copy its structure and change the part the task needs.
4. **Write the skeleton.** A `@flyc.kernel` device function and a `@flyc.jit` launcher. Compile-time
   values are `fx.Constexpr[int]`, runtime sizes are `fx.Int32`. Keep M (tokens) a runtime value so
   one kernel serves every batch.
5. **Test correctness** against the native implementation on every served shape, then against fresh
   random inputs, then under CUDA-graph replay.
6. **Profile, then tune** (`tuning.md`). Read the bottleneck (memory, latency, LDS, MFMA
   issue) before choosing a knob.
7. **Integrate** through the run's overlay, with a stock fallback for any input the kernel does not
   support.

## 4. Reference kernels (ROCm/FlyDSL `kernels/`)

| Need | Start from |
|---|---|
| RMSNorm / LayerNorm, fused residual add, fused int8 quant | `kernels/norm/rmsnorm_kernel.py`, `layernorm_kernel.py` |
| Softmax | `kernels/norm/softmax_kernel.py` |
| QK RMSNorm + RoPE (+ fp8 group quant) | `kernels/attention/qk_norm_rope_quant.py` |
| RoPE + paged KV-cache write (+ fp8) | `kernels/attention/fused_rope_cache_kernel.py` |
| FP8/INT8/INT4/BF16/FP4 GEMM, fused epilogues | `kernels/gemm/` (preshuffle GEMM), `examples/04-preshuffle_gemm.py` |
| MoE GEMM and reduce | `kernels/moe/` |
| Paged / flash / MLA / SWA attention | `kernels/attention/pa_*.py`, `flash_attn_*.py`, `mla_fwd_decode*.py`, `swa_gfx950.py` |
| All-reduce, comm | `kernels/comm/`, `kernels/monokernel/symmetric_allreduce.py` |
| L3 MegaKernel (MoE + dispatch/combine) | `kernels/mega_moe/`, `kernels/mega_moe_tp/` |
| L4 MonoKernel (resident decoder layer) | `kernels/monokernel/` (README, `glm/`, `k3/`) |
| Shared helpers (buffer ops, DPP, layouts) | `kernels/common/` |
| Teaching examples | `examples/01-vectorAdd.py`, `02-tiledCopy.py`, `03-tiledMma.py` |

GEAK's own FlyDSL cards: `perf_knowledge/operators/*/backends/flydsl.md` (rmsnorm, gemm_epilogue_fused,
grouped_gemm_moe, layout_shuffle, mla_attention).

## 5. Kernel fusion with FlyDSL

Rules and patterns are in `fusion-recipes.md`. In short:

- **Keep intermediates in registers.** The fused kernel reads the region's inputs once and writes its
  outputs once. A value that crosses a thread boundary goes through a shuffle (`gpu.shuffle_xor`,
  `ds_bpermute`) or LDS, not through HBM.
- **One owner per output.** Map the grid so each output element (and each KV slot) is written by exactly
  one lane. Skip invalid work with a runtime `if`, never with a divergent barrier.
- **Reduce once.** When two reductions share a pass (a residual sum and a sum of squares), pack them into
  one block reduction (`fx.coop.BlockReduce[fx.Float32x2, ...]`).
- **Fold scales.** Apply quant and dequant scales in f32 registers. Clamp before fp8 conversion
  (`e4m3fnuz` encodes -0 as NaN).
- **Single kernel first, then fuse.** When a member of the region is itself worth optimizing (a GEMM
  that needs tuning), tune it first. Then fuse its neighbours into its epilogue or prologue, with the
  tuned kernel as the baseline the fused kernel must beat.
- **Envelope.** Engage the fused kernel only on the shapes and phases where it was measured faster, and
  fall back to the stock path elsewhere.

## 6. Kernel optimization with FlyDSL

Knobs, limits and the profiling loop are in `tuning.md`. Checklist targets for a GEMM hot
loop: MFMA ≥ 40% of the loop, combined VGPRs ≤ 256, no spills, LDS conflicts removed by swizzle.

## 7. Debugging

`debugging.md`. First step every time: clear the cache (`rm -rf ~/.flydsl /tmp/flydsl*`,
or `FLYDSL_RUNTIME_ENABLE_CACHE=0`). A stale artifact is the most common cause of "the fix did not work".

## References
- `api.md`: API cheat sheet (decorators, types, control flow, layouts, copy, LDS, MMA, sync).
- `fusion-recipes.md`: fused-kernel patterns with skeletons.
- `tuning.md`: performance knobs and the profiling loop.
- `debugging.md`: symptoms, causes and checks.
