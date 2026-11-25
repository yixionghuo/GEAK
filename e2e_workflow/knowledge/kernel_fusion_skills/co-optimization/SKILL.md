---
name: kernel-fusion-co-optimization
description: >
  How to make a kernel fusion faster than the sum of its parts (1+1>2) instead of slower (1+1<2):
  the order in which to fuse and to optimize the kernels, the kernel-level levers that only exist
  inside a fused/persistent kernel (cross-stage prefetch, overlap, claim order, config re-tuning,
  sync-scope reduction, codegen), and how to attribute each gain. Distilled from the GEAK mode=mega
  MegaMoE EP8 work on gfx950 (8192 tok/rank 1.22x, 64 tok/rank 1.18x), cross-checked on five more
  mega fusions (Muse Spark attention block and MoE on MI355X, Muse attention-output boundary and
  Qwen3 MoE on MI300X) and on serving e2e of two of them. Use when planning or authoring any
  multi-stage fusion (GEMM->GEMM, GEMM->comm, dispatch/combine, quant->GEMM, attention block, etc.),
  when predicting its serving gain, or when shipping it as a library op.
version: 3
sources: GEAK perf_knowledge/expert_skills (megamoe_ep_mega_fusion, megamoe_small_mtpr_fusion,
  expert_workflows/mega_fusion/knowledge/*), on-card mega waves 2026-09..11; v2 adds the GEAK
  mode=mega runs and serving work packed in /sgl-workspace/ref (Muse attn 20261103, Muse attnblk and
  MoE 20261106, Qwen 20261028 and scratch 20261102, Muse 1027 e2e, aiter PR #7870); v3 adds the
  mid-fusion toolkit (floor worksheet with MI355X calibration, symptom table with numbered
  do-not-retry rows, verified ISA/metadata commands, HIP region stamps + critical-path reducer)
---

# Kernel fusion with kernel optimization: getting 1+1>2

## 0. The answer in one paragraph

Neither "fuse first" nor "optimize first" is right on its own. **Fusion removes waiting; it never
removes work.** In every measured case (six operators) the fusion itself was a small gain or a loss and most of the
speedup came from kernel-level optimization done *inside* the fused structure — optimizations that
either did not exist across a launch boundary or that the fusion had broken and had to be restored:

| case | baseline | fusion alone | after in-fusion kernel levers | share from in-fusion levers |
|---|---|---|---|---|
| EP8 MoE, 8192 tok/rank | 4.70 ms | 4.50 ms (1.045x) | ~3.84 ms (~1.22x) | ~0.66 of ~0.86 ms (~77%) |
| EP8 MoE, 64 tok/rank (graph) | ~344 µs | ~326.5 µs (1.05x) | ~291 µs (1.18x) | ~35.5 of ~53 µs (~67%) |
| Muse attention block, TP4, gfx950 | 8 launches | 0.685x (grid barriers) | 1.174x | all of it; fusion alone lost |
| Muse attention-output, TP8, gfx942 | 8 launches | 1.016x | 1.497x | ~97% |
| Muse MoE, TP4/EP4, gfx950 | 7 launches | 1.038x | 1.08x deployable | front/tail only; experts at ~91% of HBM floor |
| Qwen3 MoE EP8, gfx942 (tuned baseline) | 8 launches | 0.315x | 1.054x | all of it |
| Qwen3 MoE EP8, gfx942 (scratch baseline) | 10 launches | 0.837x | 1.25x | all of it |

In none of the six operators did fusion alone pay more than ~5%; in three it was a loss until the
stages were back at parity. Ledgers in `05`.

So the order is: **tune and measure each stage first (that is the parity bar), build the fused
skeleton for correctness while carrying every stage's tuned configuration, restore any per-stage
efficiency the fusion lost, then spend effort on the levers that only fusion makes expressible, and
finally re-tune constants for the new structure.** Plan the fusion structure *with those levers in
mind* (co-design): a structure that cannot host them caps the gain at the launch/wait savings.

## 0.1 What all six fusions agree on

The consolidated lessons. Each line holds on at least two operators (18 is from one, with its
full round history); details and numbers in the
referenced section.

**Before writing the kernel**
1. The baseline is the production path that serves today, with graph capture. Against a scratch
   chain the same Qwen3 fusion read 1.25x, against tuned library kernels 1.05x. (`01` §3)
2. Launch count is not the cost. Inside a captured graph kernels run back-to-back; removing 7
   launches saved ~1 µs where a launch model predicted ~18.5 µs, and launch-only fusion of the Muse
   MoE saved nothing. Price fusion by waits, joins and lost overlap. (`01` §3)
3. Pick the buckets from the target recipe's batch distribution. Gain follows the covered share
   of GPU time, not the kernel speedup; a recipe change can move it to ~0. (`07` §3, §6)
4. Find where the time goes first: a stage that production already runs at the hardware floor
   (expert GEMMs at ~91% of HBM) stays outside; fuse the fixed costs around it. (`01` §4.5)
5. Balance before fusing. Ragged work across CTAs (token-balanced attention, 1.22x with no fusion)
   and across ranks (EP balance, the largest MoE lever) beat any fusion lever. (`02` B11-B12)

**Building the fused kernel**
6. Expect the first skeleton below 1.0x (0.32x-1.05x in all six). Parity per stage comes first;
   only then do overlap levers count. (`01` §4.3)
7. No grid barriers: readiness edges with monotone counters from the start (4-6 µs per barrier at
   1 CTA/CU; 0.685x and 0.837x skeletons). Narrowest scope: agent inside a rank, system only for
   peer edges. (`02` C1, C9)
8. Do not rewrite a tuned stage in another DSL to fit it in (5-10% slower, or 13-19% of HBM). Reuse
   the tuned body or keep it outside. (`02` E6)
9. Design for 1 wave/SIMD at 256 registers. Every 2-wave attempt was blocked or slower (415 regs,
   491 VGPR+AGPR). Residency is a correctness invariant; claim dependent work by ticket, never by
   program_id. (`02` E4-E5)
10. Prefetch only bytes that must be read anyway, and only under a wait that exists anyway. Extra
    early weight reads lost on every operator they were tried on (four). (`02` A7)
11. Watch the in-order `vmcnt` queue: a prefetch or push issued in the wrong place drains with the
    loads it was meant to overlap; a conditional load can add `vmcnt(0)` per block. (`02` A5, F10)
12. Look for algebra that moves work off the critical path (router through the norms: +18 pt), and
    let reductions ride communication that runs anyway (split-K in the reduce-scatter). (`02` A10-A11)
13. Cross-rank: data as its own flag, push not pull, rotate peers, one publish per destination, no
    per-store drains to peer memory. (`02` B10, B14, C10-C11)

**Measuring and shipping**
14. Resolution: ~2-3% per pair in UT, +-3% throughput / +-1.5% P90 in serving. Stop micro-tuning
    after 2-3 rounds below it; change the structure or the bound. (`05` §J, `07` §5)
15. Confirm in serving by probe: per-forward GPU time by batch size, compared with a UT prediction
    weighted by the batch distribution. Explain "no gain" by the factor that went to zero. (`07`)
16. Faster kernels change timing and expose latent distributed bugs (a TP desync became a deadlock);
    check cross-rank consistency and run both arms with the same fix. (`04` trap 13)
17. A library op needs host checks that every rank evaluates the same way, collective setup and
    close, bounded spins that trap with a record, and UT of every output and every bad input.
    Host checks run only at graph capture. (`08`)
18. In the middle of a fusion, measure before choosing: the largest jumps came from a region
    timeline, a bytes-fetched check (2x over-fetch) and the ISA (a hidden `vmcnt(0)`), and a floor
    calculation stopped three rounds of tuning a region already at its floor. (`09`)

## 0.2 Scope: which parts apply to your fusion

The content has three layers with different reach:

| layer | where | applies to |
|---|---|---|
| method (order, per-edge test, parity bar, attribution, structural vs constant) | §0, §2, §4, `01`, `03` | any fusion, any hardware |
| levers for multi-stage producer/consumer fusion (claim order, readiness, queues, sync scope, residency) | `02` §B–§E, `04` | persistent / megakernel / cross-stage fusion with inter-block dependencies |
| concrete numbers and recipes | all tables, `05` | measured on six operators, gfx950 and gfx942, FlyDSL / Triton / HIP — calibration only |
| mid-fusion diagnosis (timeline, floors, symptom -> lever, ISA reading, worked example) | `09`, `tools/` | any fused kernel that is correct but not yet fast |
| serving and shipping | `07`, `08` | any kernel meant for production serving; numbers from Muse Spark on MI355X |

Tags used in `02` and `04`:
- `[AMD]` — AMD Instinct ISA/topology (multi-XCD L2, `s_waitcnt`, SGPR/`readfirstlane`, AGPR).
  Not valid on NVIDIA; check the XCD count and partition mode on other Instinct parts.
- `[gfx950]` — lowering, cache-policy bits, limits or numbers read on gfx950 (MI355X). Re-dump the
  ISA on any other target or compiler version.
- `[FlyDSL]` — FlyDSL tracing/lowering behavior. Other DSLs (Triton, CK, HIP) have their own traps.
- `[multi-rank]` — only for fusions with cross-GPU communication.
- untagged — mechanism expected to transfer; the numbers still have to be re-measured.

What to read by fusion type:

| fusion type | examples | read | mostly skip |
|---|---|---|---|
| epilogue / elementwise | GEMM + bias/act/quant, norm + quant, RoPE + KV write | §1 items 1, 3, 6; `01` §3 (ladder) and §4.4; `02` §N, §E1, §E4, §E6, §F; `03` §4–§5 | `02` §B–§D, overlap meter, all multi-rank items |
| tile-internal on-chip pipeline | flash attention, fused MLP | `01`; `02` §A, §E, §F, §G; `03` §1, §4–§5 | claim order / queues (`02` §B, §D) unless persistent |
| multi-stage persistent, single GPU | GEMM -> GEMM, quant -> GEMM -> reduce, megakernel layer | everything except `[multi-rank]` items | — |
| cross-rank compute + communication | MoE dispatch/combine, GEMM + all-reduce / reduce-scatter | everything | — |

Questions to answer before reusing any number: same vendor/arch? same DSL and compiler version?
same regime (fixed-cost / bandwidth / compute + comm)? If any answer is no, keep the mechanism and
re-measure the value. Evidence covers six operators on gfx950 and gfx942 in HIP, Triton and
FlyDSL, merged by topic in `02` and `04` (source marks `{attn}` `{moe}` `{attn-out}` `{qwen}`) and kept
per operator in `05`. The shares in §0
are those operators' results, not a general ratio.

Beyond the kernel:

| question | read |
|---|---|
| will the kernel gain show up in serving / e2e? | `07_kernel_to_e2e.md` |
| shipping the fused kernel as a library op (checks, collective setup, bounds, UT, PR) | `08_shipping_a_fused_kernel.md` |

## 1. Why 1+1<2 happens (check every item before blaming "fusion doesn't pay")

A fused kernel loses to the multi-launch form when it silently gives up something the separate
kernels had. Measured causes, largest first:

1. **Lost per-stage tuning.** The fused path runs a stage with a default config instead of the
   baseline's per-shape tuned one (tile, waves, cache policy, persistence). No overlap recovers that.
2. **Lost block-to-tile mapping.** A persistent scheduler replaces the original grid mapping that
   gave operand reuse in L2. A plain linear (m-major) claim order made fused GEMM2 *no faster than
   standalone*; the column-grouped claim order alone moved 8192 from 1.045x to ~1.155x.
3. **One resource shape for all roles.** The fused kernel has one VGPR budget, one block size, one
   LDS allocation sized by the largest role. Register unification alone can drop the cheaper role
   from 3 to 2 waves/SIMD (-33%) and no role partition fixes it.
4. **Synchronization scope too wide.** On multi-XCD gfx950 an agent/system release is an L2
   writeback; per-tile or per-CTA fences cost tens of µs (per-CTA system release +54 µs; per-block
   L2 writeback instead of write-through P2P stores +0.41 ms at 8192).
5. **Software join instead of hardware join.** A kernel boundary is a load-balanced hardware join;
   an in-kernel counter is a software max-of-N over few big resident blocks. Folding a 0.9 µs quant
   pass into the persistent kernel cost +16.5 µs (8192 uniform) to +65 µs (8192 skew).
6. **Codegen regressions.** A nested loop around a GEMM body made its live range loop-carried
   (~2.6x slower); an atomic in a loop tail shifted register allocation and faulted elsewhere;
   hoisted constant addresses spilled.
7. **Overlap machinery with nothing to overlap.** Per-tile readiness/peek/preemption on a balanced
   route adds atomics and polls with no payback (skew preemption on uniform +0.30 ms; it saves
   0.51 ms on skew — gate it on a runtime skew statistic).
8. **Whole-grid join inside one kernel.** `drain_all(A) -> barrier -> drain_all(B)` has one launch
   and the original serialization. It measures the *cost* of fusion, never a gain. Phase barriers
   left two other skeletons at 0.685x and 0.837x.
9. **A tuned stage replaced by a compiler tile.** Production library kernels (ASM / tuned Triton)
   rewritten to fit the fused body lost 5-10% or ran at 13-19% of HBM. The loss is the fusion's.
10. **Extra early reads.** Weight prefetch that adds HBM reads competes with the payload and the
    stage it overlaps (+2..+10 µs; +16% on an attention region already at its K/V floor).

## 2. Procedure (the order that produced 1+1>2)

### Step 1 — Measure and tune the stages separately (rungs before fusion)
- Profile per stage: duration, grid size, waves and tail occupancy, per-pipe utilization
  (MFMA/VALU/LDS/HBM), stall class, inter-kernel gap, exposed cross-rank wait (no-payload control).
  See `references/03_measurement_and_attribution.md` §1.
- Work the cheap ladder first and stop where the gap closes: remove work -> adjacent fusion ->
  concurrent independent branches -> graph capture -> dependent launch -> persistent fusion.
  `references/01_decide_and_order.md`.
- Record each stage's **tuned config and time per shape bucket**. This is the parity bar every
  fused stage must reach before any fusion gain is real.
- Do **not** spend much on standalone-only optimizations the fused scheduler will discard
  (e.g. standalone persistence/spatial remap is forced off inside the fused kernel; its reuse must
  be re-created by the fused claim order instead).

### Step 2 — Decide the structure per edge (with the levers in mind)
For every producer->consumer edge write: partial handover possible? idle engine during producer?
producer waves > 1? window vs per-edge sync cost; which in-fusion levers (§3) this edge enables.
Fuse only edges that pay or that enable a lever; leave others as `whole` (barrier-equivalent) or
outside the kernel. Name the op with the worst resource fit and price it; a fused kernel that
excludes it is a legitimate design. Pick one structure per **size regime** (see
`references/01_decide_and_order.md` §4).

### Step 3 — Build the fused skeleton for correctness, carrying tuned configs
- Reuse the baseline numerical bodies; change ownership, placement, queueing, publication.
- Default every unproven edge to `whole`; convert proven ones one at a time (bisectable).
- Wire each stage's tuned bucket config into the fused path through a per-bucket selector.
  **Tuned constants live in the selector, not in env switches**; only the structural on/off is a
  switch (a switch value exported at every batch size forces e.g. 8192 constants onto bs=512 and
  breaks it).
- Liveness first: whole-grid residency asserted from emitted metadata, monotone/epoch counters,
  reset-free arrivals, acyclic coordinator, path marker on every rank.

### Step 4 — Restore per-stage parity inside the fusion
Run the round loop of `references/09_mid_fusion_diagnosis.md` from here on: region timeline
(`tools/region_stamp.h`, `tools/region_timeline.py`), floor per region (`09` §3 worksheet — compute
it before the first lever), symptom -> lever table (`09` §4), one compile-time arm per round,
ISA/metadata check (`09` §5), paired A/B.
Compare each stage's in-fusion **busy** time (stage meter) with its standalone tuned time. A stage
slower inside the fusion is fixed in place before any overlap work:
claim order / grouping for operand reuse, chunk size, LDS by lifetime (max not sum), widen tiles by
the wave ratio, descriptors inside the unit frame, codegen fixes from the emitted ISA.
`references/02_in_fusion_levers.md` §B, §E.

### Step 5 — Add the levers that only fusion makes expressible
Cross-stage prefetch (consumer prologue and weights during producer), same-XCD handoff with no
release fence and L1-only invalidate, phase-staggered CTA pipeline (tail filling), third queue for
the reduction/combine, non-destructive peek preemption gated by skew, deferred/once-per-workgroup
publication. Each with a paired A/B. `references/02_in_fusion_levers.md` §A, §C, §D.

### Step 6 — Re-tune constants for the new structure, per bucket
Adopt or reject a structural lever only with its constants re-tuned for it (tile, waves, chunk,
group size, prefetch depth, reducer count). Sweep interacting constants together (claim order x
chunk), with fresh inputs per variant and fresh processes.

### Step 7 — Attribute and accept
- Same tree, fusion switch toggled, A/B/A/B: separates fusion gain from kernel-efficiency gain
  (shared-code edits also speed the unfused path — both ship, but credit them correctly).
- Report span, never Σ stage timers; report per-pair spread; rank-max; every guard (small/large x
  uniform/skew); fallback buckets reported honestly.
- `references/03_measurement_and_attribution.md`.

## 3. Lever catalog (summary; details and numbers in `references/02_in_fusion_levers.md`)

The levers that only exist inside one kernel, grouped by mechanism (start the next stage early,
move work to slack, cheaper handoff, pipeline across CTAs, share on-chip resources), with their
preconditions: `02` §0.

| family | lever | measured effect (unmarked: MegaMoE, gfx950; marks as in `02`) |
|---|---|---|
| balance | equal KV tokens per CTA across ragged requests `{attn}` | 1.22x full layer, no fusion needed |
| balance | balance expert load across ranks `{moe}` | +6.4% (full replication; partial redundancy to deploy) |
| algebra | router decomposed through the norms, early term during waits `{attn-out}` | 1.30 -> 1.46x |
| overlap | split-K half summed inside the reduce-scatter `{attn}` | 1.081 -> 1.118x |
| overlap | prefetch under a wait that exists anyway (W1 under the dispatch spin) `{qwen}` | +0.3-1.4% |
| sync | readiness edges instead of grid barriers `{attn}` `{qwen}` | barriers: 0.685x / 0.837x skeletons |
| sync | buffer stores to peers, one 128 B line per flag `{moe}` | scatter_done 24.6 -> 19.8 µs |
| comm | once-per-destination arrival publish `{qwen}` | front 33.9 -> 23.4 µs |
| comm | rotate all-gather peers; push not pull `{attn-out}` | AG 4.48 -> 3.28 µs; pull -30% |
| codegen | unconditional operand loads (no per-block vmcnt(0)) `{attn-out}` | MMA 9.4 -> 6.1 µs |
| codegen | contiguous 256 B rows for peer writes `{qwen}` | 0.869 -> 0.955x |
| overlap | issue next-stage weights (B) before the dependent A gather | -1.3 µs @64 |
| overlap | K-loop lookahead 2 (not 3/4) | -0.7 µs @64; 3/4 +1..4 µs |
| overlap | cross-tile B: claim next unit at K-step 24/28, issue its first B | -1.2 µs @64 |
| overlap | consumer issues B + metadata, then waits for readiness inside the tile | -0.7 µs @64 |
| overlap | issue B and A-scales before the barrier and the next A DMA (vmcnt order) | ~+3% @8192 |
| overlap | combine fused as an irreversible third queue | 4.62 -> 4.50 ms @8192 |
| scheduling | column-grouped claim order (GM=32) for W2 reuse in L2 | 4.50 -> ~4.05 ms @8192 |
| scheduling | chunk C=1 with that order (C=16 4.88 vs 4.09 ms) | tail quantization |
| scheduling | non-blocking payload-spin fill (CAS-claim a ready unit while waiting) | -0.7% @8192 |
| scheduling | deferred per-workgroup completion count | -0.8% @8192 |
| scheduling | skew-gated m-block interleave across peers | skew 0.58x -> 1.21x |
| scheduling | XCD-contiguous slices, consumer units on the same XCD counter | enables fence-free handoff |
| scheduling | split the last m-tile into BN/2 units | -0.8 µs @64 |
| config | per-bucket selector; constants not switches | correctness at all sizes |
| config | one NW=8 slab, BN widened by wave ratio (not two 4-wave halves) | halves cost +0.53 ms |
| config | LDS = max(role lifetimes) + shared scale storage | fits 160400/163840 B |
| sync | write-through (sc0\|sc1) stores + waitcnt instead of release fence | fence +0.41 ms |
| sync | same-XCD agent atomic, no release; L1-only invalidate | -6 µs @64 |
| sync | no launch handshake; monotone epoch; counters cleared before gate | -2.5, -7 µs @64 |
| codegen | anchor constant offsets to a runtime base (LICM spill) | spills 53->20, ~+1% |
| codegen | uniform values in SGPR (readfirstlane once) | ~+0.85% @8192 |
| codegen | vectorized epilogue (8 cols/lane) | ~+1.5% @8192 |
| codegen | continuation carried in the outer loop, no nested loop | nested ~2.6x slower |

## 4. Hard rules
1. Fusion is judged on span (first start to last end), rank-max, paired and interleaved.
2. A fused stage slower than its standalone tuned version is a defect to fix, not a fusion cost.
3. Every structural lever is evaluated with its constants re-tuned; untuned constants make a good
   lever look bad.
4. Overlap machinery is gated on the condition that pays for it (skew, multi-wave producer).
5. Synchronize at the narrowest correct scope; place producer and consumer so it suffices.
6. Read the emitted metadata (VGPR/AGPR/SGPR, spills, private, LDS, occupancy) after every
   structural change; do not trust profiler VGPR columns (units of 2 on gfx950).
7. Enabling steps (producer publishes, nobody consumes yet) are judged on function; only the
   terminal step is judged on speed, against the pre-chain baseline.
8. Do not chase "one launch" as a goal; a phase under ~1% of e2e is bookkeeping.
9. Never retry an item in `references/04_rejected_and_pitfalls.md` without a new reason.
10. Baseline = the production path that serves today (tuned library kernels), never a scratch chain;
    the same Qwen fusion read 1.25x against a scratch chain and 1.05x against tuned kernels.
11. Before writing the kernel, get the target recipe's batch distribution and pick the buckets to
    cover from it; after any image or recipe change, re-measure coverage and the per-step saving.
12. A kernel claim names its e2e factors (`07` §1); "no e2e gain" names which factor went to zero.

## 5. Files
- `references/01_decide_and_order.md` — fuse-or-not test, cheaper-lever ladder, ordering, regimes.
- `references/02_in_fusion_levers.md` — the lever catalog with mechanism, preconditions, numbers.
- `references/03_measurement_and_attribution.md` — profiling, stage meter, overlap meter, A/B.
- `references/04_rejected_and_pitfalls.md` — measured losses, hang/fault patterns.
- `references/05_case_studies.md` — per-operator gain ledgers (evidence for §0.1): MegaMoE 8192
  and 64, Muse attention block, Muse MoE, Muse attention-output, Qwen3 MoE (two baselines).
- `references/06_plan_template.md` — per-edge / per-lever plan and report template.
- `references/07_kernel_to_e2e.md` — coverage, prediction from UT x batch distribution, noise,
  recipe drift, serving constraints (Muse Spark 1027 / 1106).
- `references/08_shipping_a_fused_kernel.md` — host checks, collective setup/close, bounded spins,
  UT and PR standard for a persistent cross-rank op.
- `references/09_mid_fusion_diagnosis.md` — the per-round loop once the skeleton is correct: region
  timeline and critical path, floor worksheet with calibration numbers, symptom -> lever ->
  do-not-retry table (S1-S13 -> `02` ids -> `04` L/T/H ids), ISA/metadata commands for HIP and
  Triton, occupancy formula, Muse attention 0.685x -> 1.174x worked example, round log template.
- `tools/region_stamp.h`, `tools/region_timeline.py` — per-region end stamps in a HIP kernel and the
  reducer (region table, imbalance, critical path through grid-wide waits).
- `tools/examples/stamp_demo.hip` (demo + clock-skew probe), `tools/examples/hbm_read_bw.hip`
  (practical HBM read bandwidth for floors).
- `README.md` — directory tree, reading order, procedure diagram.
- `README_zh.md` — 中文摘要。
