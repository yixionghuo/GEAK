# 03 — Measurement and attribution

Without these you cannot tell whether a fused result is 1+1>2, 1+1<2 hidden by other changes, or
noise.

## 1. Before fusing: per-stage profile

1. **Active-CU upper bound from the trace you already have.** Per kernel
   `min(grid_blocks x blocks_per_CU, CUs)`, duration-weighted. A low bound is conclusive: tiny grids
   on the critical path pin the device; only running other work beside them helps.
2. **Waves and tail occupancy per stage:** `waves = ceil(blocks/(CUs x per_CU))`, last-wave
   utilization `(blocks mod capacity)/capacity`. The tail is invisible in kernel time and is the
   bubble fusion actually fills.
3. **Per-pipe utilization** (`rocprofv3 --pmc SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAIT_ANY
   SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT`). SQ counters aggregate all shader engines: divide by CU
   count. Engine clock under load, not boot clock. Classify: throughput-bound / latency-bound /
   launch-bound / mixed (table in `02_in_fusion_levers.md` §A diagnosis).
4. **Inter-kernel gap distribution** (median, max) — prices the launch argument only.
5. **No-payload control** — replace the transferred payload with nothing; the delta is the exposed
   communication cost, the ceiling of communication overlap. Plus an instrumented peer-wait timer
   (rank-max p95, uniform vs skew). Large wait with flat bytes = serialization (scheduling problem);
   bytes moved = transfer problem.
6. **Cross-stage cost asymmetry:** ns per 1e3 MFMA per stage; walk VALU/MFMA, LDS cycles/MFMA, bank
   conflicts. Prize = `(cost_ratio-1) x stage share of e2e`.
7. **Critical path and slack** at tile granularity (nodes = output tiles; edge iff output region
   overlaps input region; record scope and what enforces it today). An optimization that claims more
   than `e2e - critical_path` is wrong.
8. **Regime map per guard** (rows per group, bytes read once, FLOPs, link bytes -> binding bound).

## 2. Inside the fused kernel: stage meter (busy vs wait)

HIP kernels: the region-end form (`tools/region_stamp.h` + `tools/region_timeline.py`, usage in
`09` §2) gives per-region end times, imbalance and the critical path. The FlyDSL contract below adds
a busy/wait split per role.

After fusion there is one kernel record; stage timers and traces go blind. Build an in-kernel
per-workgroup phase log into the first fused candidate and keep it compiling:

- switch-gated (compiled out when off, separate JIT identity); timing always from meter-off build;
- `s_memrealtime` (constant rate), preallocated buffer `n_wg x slots x 4` u64 + overflow counter;
- one lane per workgroup writes enter/leave per role; role tagged `stage` and `kind` (`busy` |
  `wait`); roles never nest; any role time > kernel span voids the dump;
- per rank/stage: `busy_ms = Σbusy/n_wg`, `wait_ms = Σwait/n_wg`, `span_ms` = union; reduce rank-max.

Reading it:
- stage `busy` above its standalone tuned time => the stage got slower inside the fusion (tile,
  occupancy, LDS/VGPR, contention) — fix it first (parity);
- high `wait` => scheduling/ordering between producer and consumer, not math;
- never add busy times or subtract them from e2e.

**Coordination term (cheapest decomposition):** `e2e - Σ(standalone stage times)`.
Measured @8192 on one harness: scattered +0.015 ms; coarse-barrier fused floor +1.29 ms; a faithful
but slow unified drain +2.27 ms; the reference fused kernel **-0.30 ms** (wall clock below the
serial sum = real overlap). Stage compute was the same in all four; the whole difference was this
term.

## 3. Overlap meter (only if you claim overlap)

A faster fused kernel is not evidence of overlap (it also removes launches, keeps L2 warm, fills
tails). If the claim is overlap:
- `overlap_fraction = |{t: >=2 roles active}| / |{t: >=1 role active}|` **only over pairs on the
  edge under test**; emit off-edge pairs separately (a summed number read 43% on a kernel whose
  fused edge overlapped 0.03%);
- `cu_fraction` (CU-time weighted) — one lingering block can fake wall-clock overlap;
- `join_exit_spread_ticks` — tens of ticks across hundreds of blocks = serialized downstream;
- controls: negative (scattered path must read ~0), positive (forced concurrency must read what you
  built); `clock_skew_ns` across XCDs (barrier then read counter); `meter_overhead_pct`;
- no cross-rank timestamp comparison; `measured: unknown` is a valid answer, a synthesized fraction
  is not;
- overlap up with no e2e drop = contention, report it as a finding.

## 4. A/B discipline

- Arms interleaved in one script/lease (sequential drift up to 4% at small sizes); rank-max;
  report per-pair spread (a 15x wider spread at the same median = a max-of-N gate, not throughput).
- Graph-time noise ~±1 µs; < 1 µs effects need >= 5 passes with consistent sign.
- Fresh inputs and fresh output buffers per variant, then a fresh process to confirm the winner
  (a variant that skips work can read the previous variant's correct output — a BN>512 sweep
  "1.096x bit-exact" was relL2 0.708 in a fresh process).
- Prove activation: the arm exported the switch and every rank printed the path marker. A JIT cache
  is per machine, not per tree: arms must compile to distinct binaries (anchor the key, separate
  processes; sha the off-arm ISA against pristine).
- Re-time the frozen baseline each session; never copy a denominator.
- Every guard: small/large x uniform/skew; say per guard which path ran; fallback buckets reported
  with their ratio, never hidden in a geomean.

## 5. Attribution: which part of the gain is fusion?

Run all three on the same lease:
```
frozen baseline (multi-launch)                      T0
candidate tree, fusion switch OFF                   T_off   -> kernel-efficiency gain = T0 - T_off
candidate tree, fusion switch ON                    T_on    -> fusion gain = T_off - T_on
```
Edits in shared code (GEMM bodies, codegen) show in both arms: kernel efficiency, not fusion — both
ship, but the next plan must target the right lever. Report the ablation ladder when possible:
adjacent fusion -> graph capture -> concurrent branches -> dependent launch -> persistent; `T4->T5`
is what the megakernel itself is worth. Report fused time raw and minus the empty-framework cost
(run the persistent framework with no work).

## 6. Correctness and liveness gates (a fast wrong kernel is the commonest false win)

- relL2 at several sizes against the reference; deterministic repeat (bit-exact) for identical input;
- route/input-changing graph replay >= 100-1000 times; first call after a change vs a repeated call
  within a tight tolerance (stale per-launch metadata only shows on the first call; symmetric inputs
  hide it — use a moving hot source);
- smaller->larger size transition on the same instance (generation padding);
- ragged per-rank sizes; bucket switch test (interleaved sizes, graph and eager);
- path marker on every rank and launch count per call;
- bounded wall-clock timeout on every on-card run; a hang is debugged with instruments (`{observed,
  target}` dump per counter, bounded spin with failure flag, smallest repro), not guesses.
  `observed == 0` everywhere = the publish path never ran (wiring/gate), not an offset bug.

## 7. Tools in GEAK (pointers)
- `perf_knowledge/expert_skills/expert_workflows/mega_fusion/tools/mega_graph_ab.py` — paired rotated
  CUDA-graph A/B with path-marker counts and relL2 vs first arm.
- `.../scripts/multi_rank_analysis/reduce_stage_meter.py` — stage meter reducer.
- `.../tools/mega_regime.py` — regime map per guard.
- `.../knowledge/{overlap_instrument,crash_bisection,jit_arm_isolation,gfx950_lowering}.md`.
- This skill: `tools/region_stamp.h`, `tools/region_timeline.py` (HIP region stamps, critical
  path), `tools/examples/hbm_read_bw.hip` (HBM floor calibration).
