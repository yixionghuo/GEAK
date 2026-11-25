---
title: fused_moe_grouped_gemm on FlyDSL (fp4 a4w4) — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: flydsl
gens: [gfx950]
dtypes: [fp4_e2m1, fp8_e4m3_fnuz, bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-09
sources:
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/moe_kernels.py
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/test_flydsl_moe_a4w4.py
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - https://www.lmsys.org/blog/2026-05-28-mori/
  - https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
---

# fused_moe_grouped_gemm × FlyDSL (fp4 a4w4)

## TL;DR
The fp4 **a4w4 block-scaled** MoE path: `flydsl_moe_stage1` (fused gate+up, A·W1) followed by
`flydsl_moe_stage2` (down-proj A·W2), with `a_dtype="fp4"`, `b_dtype="fp4"`, per-1×32 MXFP4 weights/acts and
e8m0 scales. Stage1 can **fuse activation + requantization** in-kernel (silu_and_mul + fp4/fp8 quant + tiled
scale write), so the fp8-input intermediate is produced ready for stage2 without a separate pass. When
`b_dtype=="fp4"` the compile dispatches to `mixed_moe_gemm_2stage`. This is the path aiter FusedMoE uses for
mixed precision (A4W4) and the kind of MoE GEMM AMD reported on Kimi-K2.5. On MI355X the A4W4/MXFP4 path
cuts latency **1.6×** at concurrency 512; MXFP4 GEMMs are ≈**62%** of Llama2-70B e2e, so this is the
dominant low-bit MoE lever. Targets CDNA4 (gfx950) for fp4 MFMA. Gate on `is_flydsl_available()`.

## SOTA implementation
`flydsl_moe_stage1` infers fp4/fp8 fused-quant from `out_dtype`, runs the (optionally split-K) fp4 GEMM,
then for the split-K / gate-up-interleave fused cases calls the cached fused silu+quant module. From
`/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py`:

```python
_need_fp4 = out_dtype == "fp4"
_need_fp8 = out_dtype == "fp8"
_fuse_any_quant = _need_fp4 or _need_fp8
...
if _gui_sk_fused:                                  # gate-up-interleave split-K + fused quant
    _quant_mode = "fp4" if _need_fp4 else "fp8"
    _silu_fused_k = _get_compiled_silu_fused(inter_dim, topk, _quant_mode, gui_layout=True)
    _run_compiled(_silu_fused_k, (tmp_out.view(-1, inter_dim * 2),
        out.view(-1).view(torch.uint8), out_scale_sorted_flat,
        sorted_token_ids, num_valid_ids, token_num, num_sorted_rows,
        torch.cuda.current_stream()))
```

The mixed kernel builder exposes a `GateMode` enum
(`kernels/mixed_moe_gemm_2stage.py`): `SEPARATED` (default, two B-tile streams), `INTERLEAVE`
(weight rows interleave gate/up; pack_N=2 routes even/odd N subtiles), `MOCK_GATE_ONLY` (single B stream
over full `[0,2*inter_dim)`, requires split-K), `GATE_ONLY` (reserved). Weights/scales follow the CK
preshuffle for fp4 — per `test_flydsl_moe_a4w4.py`: `shuffle_weight(w1_qt, (16,16))`,
`shuffle_weight_a16w4(w2_qt, 16, False)`, with `moe_mxfp4_sort` for the scales.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `compile_mixed_moe_gemm1` (gate+up fp4 a4w4) | `kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm1` | gfx950; a4w4 (fp4×2), e8m0 scale | no isolated number on-box (test asserts ≥95% close, atol 1.0) | fp4 MoE stage1 |
| `compile_mixed_moe_gemm2` (down-proj fp4) | `kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm2` | gfx950; a4w4 | no isolated number on-box | fp4 MoE stage2 |
| fused silu_and_mul + fp4/fp8 quant | `kernels/silu_and_mul_fq.py::build_silu_and_mul_fq_module` | gfx950 | no isolated number on-box | split-K / interleave stage1 post-process |
| full fp4 fused-MoE GEMM (vendor) | aiter FusedMoE on FlyDSL | gfx942/950 | A4W4/MXFP4 path **1.6× latency cut @ concurrency 512 (MI355X)**; Kimi-K2.5 MI300X: up to **+162% tput, −69% TPOT, −65% TTFT** (vendor) | mixed-precision MoE block |

## Config space / knobs
From `flydsl_moe_stage1` / `flydsl_moe_stage2` and `get_flydsl_stage{1,2}_kernels` (fp4 branch).

| param | range / typical | effect | default |
|---|---|---|---|
| `a_dtype` / `b_dtype` | fp4 / fp4 (a4w4); a_dtype fp8 also valid for w4 | activation / weight precision | s1 fp8/fp4, w fp4 |
| `out_dtype` | bf16 / fp4 / fp8 | bf16 = plain; fp4/fp8 trigger fused requant + e8m0 sorted scale | bf16 |
| `tile_m`/`tile_n`/`tile_k` | s1 n∈{32,64,128} (fp4 b); s2 m∈{16,32,64,128}, n∈{128,256}, k=256 | tiling | s1 32×256×256, s2 32×128×256 |
| `k_batch` (split-K) | 1,2,4,7,14 (fp4-a stage1) | split-K → atomic partials, fused silu reduces | 1 |
| `gate_mode` | separated / interleave / mock_gate_only | gate/up strategy (`GateMode`) | separated |
| `gui_layout` (silu_fq) | False=separated, True=block-interleaved (block 16) | input layout of fused activation | per gate_mode |
| `quant_mode` (silu_fq) | fp4 / fp8 / none | output of fused activation (MXFP4 / MXFP8 e4m3fn / bf16) | inferred from out_dtype |
| `mode` (s2) | atomic / reduce | accumulation strategy | atomic |
| `xcd_swizzle` | 0 / 4 | XCD remap | 0 |

**Authoring vs. selecting.** The table above is the *dispatch* surface of the `mixed_moe_gemm_2stage`
generation. The newer on-box mxfp4 path (`aiter/ops/flydsl/kernels/mxfp4_gemm{1,2}.py`,
`mxfp4_kname.py`) states its whole config space **as the kernel name** — a candidate is a name string that
`parse_*_name` must accept, not a free tuple of ints. The facts you need before generating candidates
are:

**The name is the config object.** `mxfp4_kname.py` holds a token grammar (`_MXMOE_NUMERIC_TOKENS`,
`_MXMOE_G1_FLAG_TOKENS`, `_MXMOE_G2_FLAG_TOKENS`, `_FLYDSL_V2_GEMM2_RE`) plus a parse/build pair, and the
tuned CSV stores a config by storing its name. Change a config by building a new name and letting the
parser reject it — an unknown token raises `bad mxmoe kernel name ... unknown token`, so an illegal
config fails at parse time instead of compiling into something that merely runs. Two stage-2 families
coexist with different grammars: `parse_g2_kname_any` dispatches between the native
`flydsl_mxmoe_g2_a4w4_*` form and the v2 `flydsl_moe2_layout_a{a}_w{b}_{out}_t{M}x{N}x{K}_{epilog}[_persist][_nt][_sbm{N}]`.
`persist` and `nt` are spelled differently in each, and a token legal in one is unknown in the other.

**Stage 1 is a whitelist, not three knobs.** Pick `(BM, use_nt, inline_quant)` as one choice from
`MXFP4_G1_VARIANTS[a_dtype]`: fp4 admits `{(32,T,F),(32,F,F),(64,T,F),(64,F,F),(128,F,F),(16,T,T)}`, fp8
the same set minus `(64,T,F)`. Anything else raises `NotImplementedError` from `_assert_supported`.
Consequence to read off the set directly: tile_m 16 exists **only** with non-temporal B loads and inline
quant — the deployed decode config `16x256x256_f16in_nt` — and tile_m 128 exists **only** without
non-temporal loads. There is no `(16,*,False)` and no `(128,True,*)` to try. BK is welded: stage 1 asserts
`BN in (64,128,256) and BK == 256`, making `K % 256 == 0` a hard shape requirement, as is `(2*D_INTER) % BN == 0`.

**Four values move together with the sort block.** `sort_block_m = tile_m` in the generic 2-stage
compiler, with no stage-1 override; `compile_mixed_moe_gemm2` accepts its own `sort_block_m` but raises
unless it equals stage 2's tile_m or is an exact multiple. `native_scale_layout_for(BM, out_dtype)` is
true exactly when `BM == 16 and out_dtype == 'fp4'`, and stage 2's BM16 A-scale chunk addressing is
selected by `SBM == BM`, not by `BM` alone — so stage-1 BM, the sort block, stage-2 tile_m and stage-2
`SBM` are one decision. A mismatch is refused at runtime (`FlyDSL v2 stage2 sorting layout mismatch`).
The two scale layouts are different *shapes*, not different strides, so changing one side alone is
silently wrong rather than a crash. Read `_sbm{N}` in a name as a declaration, not a deviation:
`build_flydslv2_gemm2_name` appends it whenever `sbm` is non-zero, and every shipped GLM-5.2 name carries
an `_sbm` equal to its own `t{M}`.

**Epilogue and output dtype are one decision.** `out_dtype='fp8'` requires `epilog='reduce'`;
`out_dtype='fp4'` requires `epilog='scatter'`, which in turn requires `BM == 128`; only `bf16` may use the
atomic path. The parser rejects the rest outright (`atomic incompatible with f4out|cshuffle`) because the
atomic add *is* the topk combine, and a packed-fp4 or LDS-shuffled store has nothing to add into.
`persist` is a property of the epilogue (`epilog in ('nonatomic','nonatomic_mxfp4')`), and the XCD swizzle
is wired only into the persistent branch — an `atomic` kernel carrying an xcd token is not swizzling.

**atomic vs reduce is a different compiled kernel, not a flag.** Atomic adds each partial inside stage 2's
epilogue (`llvm.AtomicRMWOp(fadd)`); reduce leaves the output unsummed and runs a second kernel from
`moe_reduce.py`, separately `lru_cache`-compiled and separately launched. Budget the extra host dispatch.
AITER's shipped GLM-5.2 rows take `atomic` for every bucket from `M<=16` through `M257-1024` and `reduce`
(sometimes `persist`) above `M>4096`; the cutover is somewhere in between, not at one universal M.

**`inline_quant` (`f16in`) swaps the A pipeline out.** The kernel takes the bf16 hidden state as an extra
argument and quantizes per 32-element group straight into the swizzled A-LDS slot, so the global→LDS DMA
for A and the whole A-scale load are **skipped**, not supplemented. The cost moves to in-loop ALU: a
packed-u16 amax over four dwords, two DPP reduction steps, then `cvt_scalef32_pk_fp4_bf16`. It is legal
only at `BM == 16` (hence only with `use_nt=True`), is rejected with `k_wave > 1` and `BN == 64`, and is a
prerequisite for `prefetch_hidden` (`hpf`).

**Stage 1's activation costs a full LDS round trip.** MFMA leaves the gate column and its paired up column
in *different waves'* accumulators, so stage 1 barriers, writes every accumulator to an f32 LDS buffer,
barriers again, and reads the pair back before multiplying; any `k_wave` cross-wave reduction happens on
that read. Bias, if any, goes in at that one site — after the pair is reassembled, before the activation.
Pick the activation by clamping semantics, not name: `silu` clamps both before the multiply (the host
passes `+inf` for unclamped SiLU, so `swiglu_limit` is live even then), `swiglu` clamps gate from above
only and up on both sides with a 1.702-alpha sigmoid on `(up + 1.0)`, `situv2` is unclamped.

**Stage 2 weights before the atomic.** The epilogue multiplies each accumulator pair by the per-row topk
weight in f32, converts to bf16, then issues one packed-bf16x2 atomic fadd. Converting before weighting
changes the rounding; weighting after the atomic is not expressible. Budget an LDS transpose on the way:
MFMA leaves four strided rows per lane while a packed 2-wide atomic needs contiguous column pairs, so the
accumulators pass through LDS purely to be re-laned, then each lane issues four atomics 64 columns apart
to cover BN=256.

**Nothing checks your LDS budget.** No comparison of the computed `lds_bytes` against hardware LDS size
exists in either stage. The one budget constant, `LDS_2WG_LIMIT` (65 KiB), feeds `default_k_stages`, which
tries the deep pipeline, then the shallow one, and **returns the deep one anyway if neither fits** — an
over-budget tile is a link-time or occupancy surprise, not a compile error. Size it yourself: stage 1
*unions* the A region with the accumulator
(`max(k_wave*kAStages*BM*KH_TILE + kSubBlocks*(K/BK)*256, k_wave*(BM/epi_splits)*(BN+4)*4)`, `KH_TILE` =
`BK` for fp8 A and `BK/2` for fp4), while stage 2 does **not** union — its layout is sequential — so the
same tile numbers cost more LDS in stage 2. Keep `ACC_LDS_STRIDE = BN + 4`: those four dwords rotate each
four-row store group by 16 banks on gfx950 and exist to avoid a 4-way conflict. Stage 2's accumulator has
no such padding, so a store pattern copied across inherits the conflict.

**Two stage-1 families compete at the same shape.** `moe_stage1` compiles out of the generic 2-stage
builder (`mixed_moe_gemm_2stage_common.py`) with block_m=32 in every shipped row for this shape;
`mxmoe_stage1` compiles out of `mxfp4_gemm1.py` (block_m=16, its own `gemm1_grid`, its own `f16in`/`hpf`
epilogue tags). At gfx950 / 6144×256 / 257 experts / topk 9 / mxfp4, AITER's table takes `moe_stage1` at
`token=1,2` and `token>=256` and `mxmoe_stage1` in between — not a single cutover. Check which module is
live before tuning knobs inside one of them; a retune is scoped to that family's own knob space.

**A fused shared expert is a cross-stage identity contract.** Both compiles need the same
`shared_expert_id`, and it must be the last index — `compile_mixed_moe_gemm1_common` and its stage-2
counterpart each raise unless `shared_expert_id == experts - 1`. It is legal only with MXFP4 routed
weights (both stages raise `Heterogeneous B requires MXFP4 routed weights` otherwise). Read a fused call's
expert axis as routed-plus-one: GLM-5.2's 256 routed experts plus one shared is why its tuned rows key on
`expert=257`.

**Parity is cosine, not elementwise.** The shipped gate is
`logits_diff = ||x-y||^2 / (||x||^2 + ||y||^2) < 0.01` (`LOGITS_DIFF_TOL`,
`op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:90`; asserted in `op_tests/test_moe_2stage.py:570`).
Do not carry the dense family's elementwise gate over — it was abandoned for quantization reasons, not
kernel error (`op_tests/test_moe_ep.py:813`). Check NaN separately and deterministically as its own
assert: both `checkAllclose`'s err and the cosine diff can be masked by atomic-reduction noise, and a
stage-2 atomic combine over an uninitialised K-padding tail is the defect that probe was written to catch.

## Numerics / parity
fp4 (E2M1) MXFP4 with **per-1×32 e8m0 block scales**. The fused `silu_and_mul_fq` kernel computes SiLU via
`llvm.amdgcn.exp2.f32`/`rcp.f32`, takes the per-32-lane abs-max (warp shuffle-xor reduction), derives the
e8m0 scale with a fixed headroom (`_fp_headroom = 2` for fp4, `8` for fp8), then quantizes: fp4 via a
hand-coded `_f32_to_e2m1`, fp8 via `rocdl.cvt_pk_fp8_f32`. Scales are written in a tiled sorted layout
(the 6-index `d0..d5` swizzle). GEMM accumulates fp32. The a4w4 test (`test_flydsl_moe_a4w4.py`) checks
≥95% of elements close at `atol=1.0, rtol=0.05` vs a dequant reference — fp4 is lossy by construction, so
verify at the task-accuracy level, not bit-exact. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Entry points `aiter.ops.flydsl.flydsl_moe_stage1` / `flydsl_moe_stage2` with `a_dtype="fp4"/"fp8"`,
`b_dtype="fp4"`. Caller supplies moe-sorted ids, CK-preshuffled fp4 weights, e8m0 weight scales, and
(for fused requant) receives `(out, out_scale_sorted)` from stage1 to feed stage2. No env-CSV overlay; the
mixed builders are `lru_cache`d on the full shape/dtype tuple. Symbols only import under
`is_flydsl_available()` (flydsl ≥ 0.1.3).

## Pitfalls & anti-patterns
- **CDNA4 (gfx950) for fp4 MFMA** — the fp4 a4w4 path targets gfx950; do not assume it on gfx942.
- Weights and scales must use the exact CK preshuffle (`shuffle_weight`, `shuffle_weight_a16w4`,
  `shuffle_scale_a16w4`) and `moe_mxfp4_sort` layout the kernel decodes — wrong layout = silent garbage.
- `inter_dim` must be divisible by 32 (MXFP4 block) for the fused activation kernel (asserted).
- fp4 is lossy; never gate correctness on bit-exact parity — use the ≥95%-close / task-accuracy bar.
- `mock_gate_only` requires `k_batch>1`; misusing it raises `ValueError`.
- **Don't hand-pick a tile for the mxfp4 path.** Its legality lives in a name grammar plus a variant
  whitelist, and an LDS over-budget tile is *not* rejected — it fails at compile time with an allocator
  error that reads like a bug. Generate candidates from the declared space (above) and let the
  benchmark pick.
- Optional dependency + per-shape JIT cost as in the standard path.

## How to verify
```bash
pytest -sv /sgl-workspace/aiter/aiter/ops/flydsl/test_flydsl_moe_a4w4.py
# test_flydsl_stage1_a4w4 / stage2 build moe-sorted fp4 inputs and check ≥95% close.
```

## Alternatives / cross-links
[[operators/grouped_gemm_moe/backends/flydsl]] (standard fp8/bf16 2-stage) ·
[[operators/fused_moe_grouped_gemm/backends/aiter]] (DB-driven A4W4 dispatch) ·
[[operators/fused_moe_grouped_gemm/backends/ck]] (block-scale stage2) ·
[[operators/act_and_mul_silu_gelu/backends/flydsl]] (the fused silu+quant kernel) ·
[[operators/quant_fp4_mxfp]] (MXFP4 quantization) · [[operators/dense_gemm/backends/flydsl]].

**Authoring / optimizing a FlyDSL MoE GEMM `@flyc.kernel`** (write from scratch OR port ck→flydsl):
[[languages/flydsl/authoring_gemm_levers]] (tiling / LDS / XCD-swizzle / epilogue) ·
[[languages/flydsl/authoring_optimization]] (structure-first workflow) ·
[[languages/flydsl/authoring_tile_programming]] (CuTe tile model) ·
[[languages/flydsl/debugging]] (correctness / NaN / hang triage).

## Sources
- On-box: `/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py` (fp4 stage1/stage2, `_get_compiled_silu_fused`),
  `kernels/mixed_moe_gemm_2stage.py` (`GateMode`, `compile_mixed_moe_gemm1/2`),
  `kernels/silu_and_mul_fq.py`, `test_flydsl_moe_a4w4.py` — `ROCm/aiter@a6bb4993`, flydsl 0.1.5.
- Kimi-K2.5 FlyDSL fused-MoE numbers (−65% TTFT / −69% TPOT / +162% tput, vendor): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- A4W4/MXFP4 1.6× latency @ concurrency 512 (MI355X) + MoRI in-kernel EP fusion: https://www.lmsys.org/blog/2026-05-28-mori/
- MXFP4 GEMMs ≈62% of Llama2-70B e2e: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
