# Tuning FlyDSL kernels (gfx942 / gfx950)

Tune only after correctness passes and a profile shows the bottleneck. Values below are from the FlyDSL
kernel tuning guide (running example `kernels/gemm/preshuffle_gemm.py`) unless marked otherwise.

## 1. Read the bottleneck first

| Measure | How |
|---|---|
| Kernel time | `rocprofv3 --stats --kernel-trace -f csv -- <cmd>` |
| VGPR / AGPR / SGPR / LDS actually allocated | rocprofv3 `rocpd_*` tables: `arch_vgpr_count`, `accum_vgpr_count`, `sgpr_count`, `lds_size` |
| Instruction-level stalls | ATT (`advanced_thread_trace`, `att_target_cu: 1`) with `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`; `hotspot_analyzer.py --topk 15 --mode both` |
| Memory-bound behaviour | `pmc_l2_analyzer.py` |
| Roofline | `flops = 2*M*N*K`; MI300X per GCD ≈ 653 TFLOPS fp8/int8, 326 TFLOPS bf16; LDS 128 B/cycle (gfx942), 256 B/cycle (gfx950) |

Rule of thumb: GEMM with M ≤ 512 is memory/latency-bound, M > 512 compute-bound. Decode-time fused
elementwise/norm kernels are memory- or latency-bound: target bytes moved and launch count.

| Stall | Fix |
|---|---|
| `vmcnt(0)` waits | Prefetch / larger `tile_k`, async copy |
| `lgkmcnt(0)` waits | Put independent work between issue and use |
| `ds_*` stalls | LDS swizzle or padding |
| `s_nop` | Scheduler hints |

## 2. Tiles (GEMM)

256 threads = 4 waves × 64 lanes; `m_repeat = tile_m // 16`, `n_per_wave = tile_n // 4`,
`num_acc_n = n_per_wave // 16`; `MFMA_per_tile = k_unroll × m_repeat × num_acc_n × 2`.

| M | tile_m / tile_n / tile_k |
|---|---|
| ≤ 32 (decode) | 16 / 64–128 / 256–512 |
| medium | 64 / 256 / 128 |
| ≥ 4096 | 128 / 256 / 128 |
| fp4 on gfx950 | 32–64 / 128–256 / 256 |

Constraints: `tile_m % 16 == 0`, `tile_n % 64 == 0`, `tile_k * elem_bytes % 64 == 0`, `tile_k` divides K,
double-buffered A `2 × tile_m × tile_k × elem_bytes` fits LDS (64 KB gfx942, 160 KB gfx950).

## 3. Occupancy and registers

- `occupancy = min(vgpr_limit, lds_limit, sgpr_limit, HW_MAX)`.
- gfx942/gfx950 arch + accum VGPRs share 512: ≤ 64 → 8 waves, ≤ 128 → 4, ≤ 170 → 3, ≤ 256 → 2,
  ≤ 512 → 1, above → spill.
- FP8 GEMM VGPR estimate: accumulators `m_repeat × num_acc_n × 4` + B tile `k_unroll × 2 × num_acc_n × 2`
  + 10–20 for addressing.
- Steer with `waves_per_eu` (compile hint). Do not force `accum_vgpr = 0` (`maxnreg` was removed: it cost
  about 4.5×).
- Small fused kernels (one wave per block) often run best at `waves_per_eu = 8` (as in
  `qk_norm_rope_quant.py`).

## 4. LDS: double buffering, swizzle, padding

- `lds_stage = 2` (ping/pong A) by default; `lds_stage = 1` can help a spill-bound tile (`num_acc_n = 8`)
  and hurts others.
- Banks: 32 (gfx942), 64 (gfx950); `bank = (byte_addr / 4) % num_banks`. A 128 B stride is a full conflict
  on gfx942 and 2-way on gfx950: re-check the swizzle width when porting.
- XOR swizzle `col_bytes ^ ((row % k_blocks16) * 16)`, applied **identically** on the global→LDS write
  and the LDS→VGPR read. Alternative: pad 1–4 elements per row. gfx950 also has `DS_READ_*_TR_*`
  transpose reads.

## 5. Prefetch and async copy

- Runtime loop with `init=` and typed bounds (`fx.Int64`) carrying the next tile; compute the current tile
  while the next one loads. Build LDS views at the kernel top (SSA dominance).
- `use_async_copy=True` (`raw_ptr_buffer_load_lds`) bypasses VGPRs: best for `tile_m ≥ 128`; DMA width
  16 B on gfx950, 4 B on gfx942. gfx1250 uses TDM.
- Skip prefetch on loops that are already memory-bound, or at 1-wave occupancy where it spills.
- Move next-phase loads into barrier-wait regions.

## 6. Loads and stores

- Coalesced `buffer_load_dwordx4` over full 64 B lines (`BufferCopy128b`, `vec_width=4`).
- Non-temporal loads for streaming inputs larger than the LLC (`rmsnorm_kernel.py`, `NT_MIN_WORKING_SET`).
- Wave-uniform values (positions, slots, scales) via `s_buffer_load` into SGPRs.
- `xcd_swizzle` spreads blocks across XCDs to fix HBM channel imbalance.

## 7. MFMA scheduling

- Hints `sched_mfma(N)`, `sched_dsrd(N)`, `sched_dswr(N)`, `sched_vmem(N)`, fence `sched_barrier(0)`.
- Standard group: 1 vmem → MFMAs → 1 ds_read → MFMAs; LDS writes at the tail.
- `enable_scheduler=True` gives +8–10% on async fat tiles; turning it off can win for f16/bf16 tiles with
  `num_acc_n ≤ 2`. Measure both.
- FP8: K64 micro-step = 2 × K32 MFMA; gfx950 adds MFMA-scale instructions.

## 8. Epilogue

Direct store by default; CShuffle (LDS remap) for `tile_n ≥ 128`. Fuse bias/activation/scale/quant here
rather than in a separate kernel (`fusion-recipes.md` §4).

## 9. Autotune and measurement

- `flydsl/autotune.py` is a Triton-style autotuner; shipped examples: `kernels/norm/rmsnorm_autotune.py`,
  `softmax_autotune.py`.
- Benchmark with a compiled handle, 60–80 iterations, median of 7; compare runs with
  `scripts/compare_benchmark.py`; bisect regressions with the repo's `bisect-perf-regression` skill.
- Checklist: MFMA ≥ 40% of the hot loop, combined VGPRs ≤ 256, no spill, no LDS conflicts.

## 10. Repo skills to load for deeper work

`gemm-optimization`, `lds-optimization`, `prefetch-data-load`, `kernel-trace-analysis`,
`capture-kernel-trace`, `isa-resource-diff`, `bisect-perf-regression`, `oob-detection`
(ROCm/FlyDSL `.claude/skills/`, also installable as the `flydsl` Claude Code plugin).
