# Fused-kernel recipes in FlyDSL

Each recipe states what it fuses, how work maps to the grid, and the FlyDSL idioms it relies on. The
shipped kernels named here are the starting point. Copy their structure rather than re-deriving it.

## 0. Rules for every fused kernel

1. **Region boundary.** List the tensors that cross the region (inputs read, outputs written, KV/cache
   slots, scales). Everything else stays in registers or LDS. The fused kernel's correctness is judged
   at that boundary only.
2. **Grid ownership.** Pick the grid from the outputs: one block per (row) or per (head, token). Each
   output element and each cache slot has one writer. Extra blocks (e.g. GQA: more Q heads than KV
   heads) do only the work they own, behind a runtime `if`.
3. **One pass where possible.** Load each input once into registers (`in_local.append(vec)` in
   `range_constexpr`) and reuse it in the second phase after a reduction, instead of reloading.
4. **Reductions inside one wave when the row fits.** A row of ≤ 64 × VEC elements needs no LDS and no
   barrier (`fx.coop.warp_reduce`). Larger rows use `BlockReduce` with LDS storage.
5. **Precision.** Compute in f32; convert at the store. Keep the native op order where the reference is
   bit-exact and the task wants bit-exactness; otherwise document the tolerance.
6. **Stock fallback.** The overlay engages the fused kernel only inside its measured envelope (phase,
   M range, dtype, layout) and calls the original callable otherwise.

## 1. Residual add + RMSNorm (+ quant)

Reference: `kernels/norm/rmsnorm_kernel.py` (`fused_add_rmsnorm`, quant variants).

- Grid: one block per row (`grid=(M, 1, 1)`, `block=(BLOCK_THREADS,)`); small N (≤ 2048) uses a
  multi-row kernel where a lane group of `THREADS_PER_ROW` handles a row and reduces with
  `fx.coop.warp_reduce(x, ADD, width=THREADS_PER_ROW)`.
- Loads: `make_buffer_tensor` + `logical_divide(row, make_layout(VEC_WIDTH, 1))` + `BufferCopy128b`
  (8 × bf16 per lane). Tail elements use a scalar path.
- Pass 1: `x = in + residual` (store the sum if the residual output is needed), accumulate `x*x`.
  Pass 2: `y = x * rsqrt(sumsq / N + eps) * gamma`.
- Quant: block max of `|y|` (`BlockReduce(..., MAX)` after a `gpu.barrier()`), `scale = max / qmax`,
  guard `scale == 0`, store `y / scale` packed.
- Large M on gfx95: a non-temporal input path (`BufferCopy128b(cache_modifier)`) once the working set
  exceeds the LLC; dispatch between two compiled variants from a `@flyc.jit` by `m_in`.

## 2. QK norm + RoPE (+ fp8 group quant)

Reference: `kernels/attention/qk_norm_rope_quant.py`.

- Grid: `(H + 1, tokens)`, one wave per block; `bid_x < H` handles a Q head, `bid_x == H` the KV row.
  Grid Y is capped at 65535: the wrapper loops over token chunks.
- Each lane owns `VEC = D // 64` elements (8 bf16 at D = 512). The NOPE part passes through after the
  norm; the last `RD` dims are rotated by lanes `NOPE // VEC` onward.
- Norm: `sq = warp_reduce(sum x², ADD, width=64)`, `rstd = rsqrt(sq / D + eps)`.
- RoPE (GPT-J pairs): `e*c - o*s`, `e*s + o*c`; `cos/sin` row = `fx.slice(cos_buf, (pos, None))`, one
  8-byte `BufferCopy(64)` load per lane.
- fp8: per-group abs-max via `warp_reduce(MAX, width=group/8)`; clamp tiny negatives to +0 before
  `cvt_pk_fp8_f32`; only group-leader lanes write scales, under `scf.if` (a masked store faulted on gfx950).
- Compile with `waves_per_eu = 8`; fold the per-token offset into the descriptor base so offsets stay i32.

## 3. RoPE + paged KV-cache write (+ fp8 cache)

Reference: `kernels/attention/fused_rope_cache_kernel.py`.

- Grid: `(max(QH, KH), T)`, one wave per block; Q work if `head < QH`, K/V work if `head < KH`.
- `VEC_WIDTH = ceil(D / WARP_SIZE)`; load `cos/sin` once per block, shared by Q and K.
- Rotary partner from a cross-lane `ds_bpermute` (`pair_lane = tid ^ vecs_per_half`), not a second load.
- Cache slot: `slot = slot_mapping[t]` (skip `slot < 0`), `page = slot // block_size`, `off = slot % block_size`.
  Load V late, right before its store (shorter SGPR liveness).
- Flash layout `[nb, bs, KH, D]`: vector store into `fx.slice(cache, (page, off, head, None))`.
  Non-flash (ATOM) layouts: element stores with computed offsets.
- fp8 cache: `rcp(scale)` once, scale in f32, pack 4 values per i32 with two `cvt_pk_fp8_f32` calls.
- Mark token-indexed tensors `mark_layout_dynamic(leading_dim=0)` so one kernel serves every T.

## 4. GEMM + epilogue (bias, activation, dequant scale, quant, split-K combine)

Reference: `kernels/gemm/` preshuffle GEMM; GEAK card `perf_knowledge/operators/gemm_epilogue_fused/backends/flydsl.md`.

- Built-in epilogues: `epilogue="bias" | "bias_relu" | "bias_silu" | "bias_gelu"`.
- Accumulator row map (MFMA 16×16): `row = bx_m + mi*16 + lane_div_16*4 + ii`.
- Row epilogue (default) for small tiles; CShuffle epilogue through LDS for `tile_n ≥ 128`
  (`e_vec = 4 if tile_n % 128 == 0 else 2`), with a global store or an atomic (split-K).
- **Single kernel first, then fuse.** Tune the GEMM alone (tile sizes, `lds_stage`, scheduler,
  `xcd_swizzle`, epilogue mode) and accept it through the gate. Then fuse the following op (quant, scale,
  activation, residual) into its epilogue, keeping the tuned tile config. The fused kernel must beat
  "tuned GEMM + separate op", not "stock GEMM + op".

## 5. Activation + gate (SwiGLU / GeGLU) + quant

Pattern: elementwise over `[M, 2N]` → `[M, N]`.
- Grid: one block per row, or tiles of `BLOCK_DIM × VEC_WIDTH` along N; read gate and up halves with
  the same per-thread `logical_divide`.
- `out = act(gate) * up` in f32 (activation helpers in `kernels/common/act.py`), then convert or quantize
  as in recipe 1.

## 6. Router / small GEMV + top-k + reduce

Pattern: decode-time router logits (M ≤ ~150 rows, E experts), top-k, normalized weights.
- One block per row (or a lane group per row for small E); logits via a GEMV (MFMA for wide K, otherwise
  vector FMA + `warp_reduce`), top-k by repeated wave max + mask, softmax over the k values in registers.
- Before fusing, measure the GEMV part alone: when the region's time is mostly the GEMV, fusing the
  surrounding small kernels saves little (a measured router region of this shape gained only 1.02×).
  Optimize the GEMV first.

## 7. Larger fusions (L3 MegaKernel, L4 MonoKernel)

Do not start from an elementwise recipe. Use the resident designs in `kernels/mega_moe/`,
`kernels/mega_moe_tp/` and `kernels/monokernel/` (README: resident grid schedule, tagged-pair mailbox
protocol, phase-overlaid LDS arena). Model-specific schedules stay with the model; compiler-sensitive
hot-loop code is not refactored without a measurement.

## 8. Checklist before handing a fused kernel on

- Correct at the region boundary on every served shape, on fresh random inputs, and under graph replay.
- A negative control (a perturbed output) fails the test.
- Timed with a compiled handle against the current baseline (which may already contain tuned members).
- Envelope recorded: phases and M range where it engages.
- `waves_per_eu`, VGPRs and LDS read from the profile, with no spill.
