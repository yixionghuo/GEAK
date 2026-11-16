# Debugging FlyDSL kernels

## 1. Always first

1. Clear the cache: `rm -rf ~/.flydsl /tmp/flydsl*`, or run with `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
   Also clear any `functools.lru_cache` around the builder (`build_fn.cache_clear()`).
2. Restart the Python process after editing a helper or rebuilding FlyDSL: compiled handles never see
   Python changes.
3. Dump what the compiler saw: `FLYDSL_DUMP_IR=1`, `FLYDSL_DEBUG_AST_DIFF=1`.

## 2. Symptoms

| Symptom | Likely cause |
|---|---|
| All NaN | `-inf - (-inf)` in a softmax, divide by zero, uninitialized buffer, fp8 `e4m3fnuz` -0 (clamp tiny negatives to +0) |
| All zeros | Wrong output address or stride, output never written, wrong partition slot |
| > 50% mismatch | Layout or grid mismatch (grid smaller than the work), wrong thread-value mapping |
| 1–5% mismatch | Quantization scale (per-row vs per-tensor), a scale applied twice, fp8 rounding |
| Compile error | Type mismatch, loop-carried state type, `range` vs `range_constexpr`, variable defined only inside a branch |
| `Mismatched ranks in intTupleZip2By` | Wrong arguments to `make_fragment_A/B/C`: build fragments from `thr_mma.partition_*` (deck, tiled-MMA case) |
| `coordinate leaf 0 must be i32` | gfx1250 TDM: a byte offset passed where a tile index is expected |
| SSA dominance error in an epilogue | An LDS view created inside a loop: create views at the kernel top |
| `init=` ignored, loop unrolled | Loop bounds were Python ints: use `fx.Int64(...)` |
| Hang | Infinite loop (`step = 0`, unsigned `stop < start`), a barrier not reached by all threads, OOB access, TDM warp layout ≠ `num_warps` |
| OOB fault on gfx950 | Masked store of scales: use a runtime `if` (scf.if) instead |

## 3. Numerical mismatch workflow

1. Clear caches.
2. All-ones inputs. Pass → layout is right, the bug is data-dependent.
3. Smallest configuration (one partition, one block, one wave).
4. Host prints: shapes, strides, NaN/inf counts, min/max of intermediates; fill outputs with a sentinel
   (e.g. -999) before launch and check it was overwritten.
5. Compare intermediates against the reference one at a time.
6. Hand-trace tid 0's addresses; check MFMA operand order (LHS → M, RHS → N).
7. `fx.printf` under a runtime `if lane == c0:`.
8. Tolerances: about 0.03 max error for fp8 against a bf16 reference is expected; bf16 elementwise
   fusions should be within bf16 ulps of the native op order.

## 4. GPU recovery

`amd-smi monitor`; reset a hung GPU only if you own it: `sudo amd-smi reset -G -g <index>`. In a shared
run, report the hang instead of resetting.
