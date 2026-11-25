# 05 — Case studies: where the gain actually came from

Operator: MegaMoE V2, EP8 on one gfx950 node; 384 experts (48/rank), top-k 6, hidden 7168,
inter 3072; MXFP8 activations, MXFP4 weights. Baseline = 4 launches per call: quantize, stage1
(dispatch + GEMM1), stage2 (GEMM2 + weighted P2P), combine.

## A. 8192 tok/rank (compute + communication regime), two-launch persistent kernel

Structure: quantize stays separate; one persistent grid runs dispatch, GEMM1 stripes, GEMM2 units,
P2P publication and combine as three queues in one CTA state machine.

| step | rank-max e2e | type |
|---|---|---|
| frozen 4-launch baseline | ~4.70 ms | — |
| stage1+stage2 persistent, combine separate | ~4.62 ms | fusion |
| + combine as third queue (m-major GEMM2 claim order) | ~4.50 ms (1.045x) | fusion |
| + column-grouped GEMM2 claim order GM=32, C=1 | ~4.05-4.07 ms (~1.155x) | in-fusion scheduling (restore + exceed reuse) |
| + 3 GEMM1 codegen items (last-K anchor, 8-col SiluQuant, SGPR row base) | 4.12 -> 3.93 ms (-4.6%) | codegen [shared] |
| + non-blocking payload-spin fill | -0.7% | in-fusion scheduling |
| + deferred per-workgroup GEMM2 completion | -0.8% | in-fusion sync |
| + combine memory order (no acquire, cache-19 loads, cache-2 store) | -0.5% | in-fusion sync |
| **best known** | **~3.84 ms (~1.22x)** | |

Observations:
- Fusion itself: ~0.20 ms. In-fusion kernel levers: ~0.66 ms.
- G1/G2 overlap stayed ~3% throughout; the claim order gain was GEMM2 efficiency (span 2.05 ->
  1.36 ms), not overlap.
- Per-workgroup busy sum ~= e2e in the best build: near work/bandwidth bound; remote P2P only
  ~0.11 ms; open lever = two-deep W2 prefetch in the GEMM2 K loop.
- Skew route needed the gated m-block interleave (0.58x -> 1.21x).
- Single costs that erase the whole fusion gain: lockstep halves +0.53 ms, per-block L2 writeback
  +0.41 ms, ungated skew preemption +0.30 ms, quant work-stealing +0.30 ms.
- A certified pinned-skill GEAK run delivered 1.136x at 8192 (2026-11-16) and 1.155x (v11).

### A.1 How each step was decided (8192)

What was measured, what it was read as, and what was done next. Reconstructed from the Skill
ledgers, the oracle source comments and the on-card investigation notes of 2026-09.

| # | measurement | reading | decision |
|---|---|---|---|
| 1 | standalone stage1 ~2.6 ms, stage2 + combine ~2.1 ms, sum ~4.70 ms; scattered e2e 4.735 ms (sum + 0.03) | launches cost nothing; the prize is only the joins between stages and folding combine. Expect ~0.2 ms | fuse with per-stage readiness; set expectations at ~1.05x for fusion alone |
| 2 | fused builds with a coarse GEMM1 -> GEMM2 barrier: e2e - Σstandalone stages = **+1.29 ms**; a unified drain **+2.27 ms**; the reference fused kernel **-0.30 ms**. Stage compute identical in all of them | the loss is coordination machinery (claim atomics on one line, nested loop spill, extra carried state), not per-stage work and not missing overlap | fix machinery: sharded 64 B heads, continuation carried in the outer loop, no nested loop (B3, D, F5) |
| 3 | per-m-tile fine readiness / peek on the uniform route: 7.5-8.1 ms vs the 6.0 ms coarse floor of the same lane; reference comment: preemption buys 0.51 ms on skew, costs 0.30 ms on uniform | balanced route: all GEMM1 tiles finish together, the early-start window is ~0; the machinery is pure cost | gate preemption on a skew statistic (B8); uniform runs drain-then-GEMM2 in one loop |
| 4 | combine as a separate launch 4.62 ms | the combine join is the remaining exposed wait | combine as the third queue -> 4.50 ms (1.045x) |
| 5 | fused GEMM2 span ~2.05 ms ≈ standalone stage2 ~2.0 ms, G1/G2 overlap ~3% | GEMM2 lost the W2 reuse the original grid mapping gave (S1, not an overlap problem) | sweep claim order on fresh inputs: GM 1/8/16/32/64 -> GM=32 (B1) -> ~4.05 ms |
| 6 | chunk C=16 inherited from the m-major order | constants tuned for the old structure | sweep chunk with the new order: C=1 (B2) |
| 7 | skew guard 0.58x with the new order | one hot expert's rows are laid out by source rank: consecutive m-blocks hit one peer link | skew-gated m-block interleave (B5) -> 1.21x |
| 8 | metadata: 75 VGPR spills, 272 B private; ISA: a constant ping/pong offset hoisted and reloaded from scratch per tile, `readfirstlane` loops on a uniform row base in the G1 K loop; stage meter: SiluQuant epilogue 245 µs per WG | codegen losses in shared GEMM1 code (S4) | three codegen items (F1-F3) -> -4.6%, spills 31; credited with the fusion-off control |
| 9 | stage meter: G1 CTAs waiting on dispatch payload | idle consumer with nothing issued (S2) | non-blocking fill of a ready G2 unit by CAS (B7) -> -0.7% |
| 10 | per-unit drain + barrier + atomic on the G2 hot path while a coarse pre-combine barrier exists anyway | publication cost with no consumer benefit | deferred per-workgroup count (C6) -> -0.8% |
| 11 | combine 89-96 µs per WG, acquire + invalidate per item | sync scope wider than needed | cache-19 loads without acquire, cache-2 store (C4) -> 83-88 µs, -0.5% |
| 12 | per-WG busy sum ≈ e2e; remote P2P ~0.11 ms; G2 K loop waits `vmcnt(2)` at every BK256 back edge | near work / bandwidth bound; one latency lever left | open: two-deep W2 prefetch (A2) |

What to copy: steps 2 and 5 were decided by a subtraction (e2e minus the sum of standalone stages;
fused stage span minus standalone stage), not by trying levers. Step 3 is the costly wrong turn:
several rounds built fine readiness for a route whose window was ~0, until the coordination term
and the reference's own A/B showed it.

## B. 64 tok/rank (weight-streaming regime), single persistent kernel

Floor = read every expert weight once (~1.68 GB/rank, ~244 µs at 6.9 TB/s). Graph time:

| step | graph µs | delta | lever class |
|---|---|---|---|
| baseline, 4 launches | ~344 | | |
| two kernels (stage1 + fused quant; GEMM2 + fused combine), row clamps | ~329 | -15 | fusion |
| single kernel, handshake still present | ~326.5 | -2.5 | fusion |
| no launch handshake | | -2.5 | sync |
| counters cleared before gate, first claim overlaps plan wait | | -7 | sync / overlap |
| plan word with num_valid + per-XCD acquire | | -1.5 | sync |
| L1-only invalidate after GEMM2 ready | | -6 | sync (same-XCD) |
| reducer L1-only invalidate | | -1 | sync |
| gate without acquire, epoch read sc0\|sc1 | | -2.5 | sync |
| no srcmap pad rows, GEMM2 row mask | | -1.5 | remove work |
| owner multi-lane fast path | | -1 | front end |
| GEMM2 setup hoisted before plan wait | | -1.7 | overlap (prologue) |
| combine reducers 64 -> 256 | | -1.5 | tail |
| payload dedup per (token, destination rank) | | -1.0 | remove work |
| GEMM2 tail m-tile as BN/2 units | | -0.8 | tail |
| early route counts | | -1.4 | front end |
| ready counters one per 64 B line | | -0.8 | sync |
| GEMM1 B before A gather | | -1.3 | overlap |
| GEMM2 ready-wait inside tile after B issue | | -0.7 | overlap |
| A-scale issued first | | -0.4 | overlap |
| GEMM1 lookahead 1 -> 2 | | -0.7 | overlap |
| cross-tile B (claim at K-step 24/28) | | -1.2 | overlap |
| GEMM2 B half-tile ring of 3 | | -0.4 | overlap |
| **final** | **~291** (official 1.174x) | | |

Grouped: fusion ~17.5 µs; sync-scope/handshake ~21.3 µs; overlap/prefetch/prologue ~6.4 µs;
remove-work / front end / tail ~7.2 µs. The sync items only exist because the kernel is fused *and* laid out
XCD-local (GEMM1 slices and their GEMM2 units on the same XCD); without that layout the same single
kernel was 75-100 µs slower.

Also required (correctness): fix the baseline LDS `work`/`has_work` race on every path.
Remaining: front end -> GEMM1 start ~20 µs (P2P floor ~11-12 µs); per-XCD GEMM1 time varies 36-40
µs with the slow XCD changing per run (largest item).

### B.1 How the steps were decided (64)

Grouped by phase; reconstructed from the Skill's ledger, timeline and rejected list (per-step
measurements were not all archived).

| # | measurement | reading | decision |
|---|---|---|---|
| 1 | floor: 48 experts x 34.0 MB = 1.63-1.68 GB -> 233-244 µs; baseline ~344 µs, 4 launches | ~100 µs above the weight-read floor; math (~10 µs) is not binding | weight-streaming regime: one persistent kernel, read every byte once, attack fixed costs (`09` §3.4) |
| 2 | a single kernel that kept a release per GEMM1 tile / per exit ticket landed 75-100 µs above the ledger's single-kernel step (~326.5 µs) | fence scope: each agent/system release is an L2 writeback, and XCD-interleaved GEMM1 slices forced cross-XCD handoffs (`02` C1) | XCD-contiguous slices + same-XCD handoff, no release fence, L1-only invalidate (B4, C2) |
| 3 | fixed sync costs per launch (handshake, counter clears after the gate, plan acquire, gate acquire) are the same at 64 tokens as at 8192 | at small size fixed costs are a large share (E7) | remove them one by one: ~21 µs in total (C5, C8) |
| 4 | timeline: front end -> GEMM1 start 20-21 µs vs a P2P write floor of 11-12 µs | front-end latency, not bandwidth | dedup per destination, early route counts, owner fast path, setup hoisted before the plan wait: ~5 µs (B10, A6) |
| 5 | GEMM1 + GEMM2 at ~1.12x of the HBM ideal; every extra early read +1..+10 µs | near the byte floor (S6): only reordering of required bytes can pay | B before A, lookahead 2, cross-tile B, half-tile ring: ~4.7 µs (A1-A4); stop adding prefetch (L1, L4) |
| 6 | combine ~4 µs after the slowest rank; each XCD's last m-tile drains alone | tail (S5) | BN/2 split of the last m-tile, reducers 64 -> 256 (B6) |
| 7 | per-XCD GEMM1 36-40 µs, slow XCD changes per run; cross-XCD stealing re-reads A (+1.3..+8 µs) | largest remaining item; the obvious balancing lever is closed | left open |

## C. Generic (no expert knowledge) GEAK run at 64 tok/rank, for contrast

A generic mode=mega search (no recipe) reached 1.09-1.13x in ~18-20 rounds (delivered 1.0928x on
2026-11-15, 2 launches vs 4). It found fusion + some in-fusion levers but not the XCD-local layout
that unlocks the fence-free sync items — the largest single group above. Lesson: the structure must
be chosen with the sync/locality levers in mind; they cannot be bolted on afterwards.

## D. What did not transfer across regimes
- The single-kernel 64 recipe asserts at 128/256 (its stage2 block_m must equal stage1 sort_block_m)
  and its one-unit-per-CTA / poll-per-tile tricks scale with T.
- The two-launch 8192 structure lost at 64/128 (0.90x / 0.877x) and its 8192 tuning was behind the
  older build below 2048 — constants must be re-tuned per size.
- Boundary between the families measured at ~200 tok/rank (cap 256: 168 tok -> single kernel wins,
  256 tok -> two-launch 1.035x).

---

# Other operators

Five more mega fusions run with the same GEAK mode=mega workflow, 2026-10..11. Same method,
different operators, GPUs and DSLs. Read them for the pattern, not the numbers.

| case | GPU | baseline launches | fused launches | R1 (fusion alone) | best measured | largest single lever |
|---|---|---|---|---|---|---|
| E. Muse Spark attention block, TP4 verify | MI355X gfx950, HIP | 8 (9 full layer) | 1 | 0.685x | 1.174x (F56 1.25x, S56 1.10x) | token-balanced attention split (no fusion needed) |
| F. Muse Spark MoE block, TP4/EP4 verify | MI355X gfx950, Triton + HIP | 7 | 5-6 | 1.038x | 1.08x deployable, 1.175x with full expert replication | dynamic EP balance (+6.4%, not deployable) |
| G. Muse attention-output boundary, TP8 | MI300X gfx942, HIP | 8 | 1 | 1.016x | 1.497x | router decomposed through the norms (+18 pt) |
| H. Qwen3-235B MoE EP8, tuned aiter/mori baseline | MI300X gfx942, Triton + HIP | 8 | 2 | 0.315x | 1.054x | hierarchical once-per-destination publish |
| I. Qwen3-235B MoE EP8, scratch Triton/mori baseline | MI300X gfx942, Triton | 10 | 1 | 0.837x | 1.21-1.28x | tile-contiguous weight streams + per-T buckets (+9.4%) |

Round-by-round reasoning (measurement -> reading -> decision) is in `09` §6 for E. For F-I only
the ledgers below were available when this version was written; their run packages
(`/sgl-workspace/ref`) are needed to add the same columns.

Common result: **fusion alone never paid** (0.32x to 1.04x). Every gain above came from levers built
after the skeleton existed, and the largest levers were often not fusion-only.

## E. Muse Spark attention block, TP4, gfx950 (one persistent HIP launch)

Block per layer at DFLASH target verify (8 tokens per request, M = 8 x batch): FP8 QKV GEMM ->
split-K reduce -> QK norm / RoPE / KV write -> [full layer: copy] -> verify attention -> merge /
o-norm / gate -> o_proj -> TP all-reduce -> boundary norm. Grid 256 x 256 threads, 1 CTA/CU,
IPC reduce-scatter / all-gather with data-as-flag. Guards F (full) / S (sliding) at M16..64.

| R | step | score (geomean F56, S56) | lever class |
|---|---|---|---|
| 1 | token-balanced FlyDSL attention (equal KV tokens per CTA), still 8 launches | 1.106 (F56 1.22x, S56 1.00x) | work distribution, no fusion |
| 2 | one launch, phases separated by grid barriers | 0.685 | fusion (cost) |
| 3 | bring every region to frozen parity | 0.859 | restore |
| 4 | readiness counters replace barriers, Wo LDS prefetch | 0.937 | sync / overlap |
| 5 | in-kernel attention at the balanced rate, group_done merge | 1.064 | restore |
| 6 | merge units on CTAs without an attention item; bf16 partials with fp32 max/denominator (half the partial bytes); Wo measured not to be the merge cost | 1.081 | scheduling / remove bytes |
| 7 | o_proj split-K x2 with the K half summed inside the RS reduce; single-round sliding merge | 1.118 | scheduling |
| 8 | deeper per-CU K/V pipeline, warm sliding page ids | 1.156 | overlap |
| 9 | 2 waves/SIMD (needs 415 regs > 256) or cross-step SWP | 1.149 | dead end |
| 10-11 | defer the gate GEMM into idle CTAs during attention | hang / S56 +16% slower | dead end (HBM contention) |
| 12 | weight loads at kernel entry, lane-parallel segment table, `v_mfma_f32_16x16x128_f8f6f4`, epilogue prefetch | **1.174** | codegen / overlap |

Observations:
- Each grid barrier cost 4-6 µs at 1 CTA/CU; 4 barriers ~18 µs on S56. Readiness edges are mandatory.
- Parity was reached only at R5; all gain after that came from in-fusion levers (R6-R12, +10%).
- The full layer is bound by K/V reads (~89 µs load+convert floor at F56). Fusion saves the fixed
  cost around it (launches, joins, o_proj+comm overlap), so the gain shrinks as context grows.
- Serving (1027 image, InferenceX AgentX, together with F): verify step 6-16% faster at batch 3-8,
  e2e +2.2..+7.9% throughput and +9.6% / +15.2% P90 interactivity at c10 / c16. See `07`.

## F. Muse Spark MoE block, TP4/EP4, gfx950

7 launches: router+pooler GEMM, routing epilogue, scatter, expert GEMM1, GEMM2, finalize+all-reduce,
unpool GEMM. M16..64.

| R | step | M56 | lever class |
|---|---|---|---|
| 1-2 | one-launch front: replicated router + this rank's 1/4 pooler slab, IPC exchange, routing / norm / MXFP4 quant / scatter | 1.041 | fusion + remove replicated work |
| 3 | prefetch w13 during the front | 1.044 (arm negative) | dead end: 0.31 µs/MB cost vs 0.12 µs/MB saved |
| 4 | dynamic EP balance, every rank holds all 256 experts | 1.108 | load balance (+5.1 GB/rank/layer: not deployable) |
| 5 | fold the unpool slab GEMM into finalize, Wu prefetch | 1.125 (below rule) | tail is xGMI-bound, not launch-bound |
| 6 | G1+G2 in one dependent-grid launch ordered by program_id | deadlock | residency not guaranteed by program_id |
| 7-8 | G1+G2 merged with atomic ticket claim | 1.136 | scheduling |
| 9 | IPC stores as `ptr_range` buffer stores, one 128 B line per flag | 1.164 (scatter_done 24.6 -> 19.8 µs) | sync / codegen |
| 11-12 | assign-once balanced scatter | 1.169-1.175 (below rule) | front micro-latency, saturated |

Observations:
- Expert GEMMs were ~91% of the HBM floor and ~150 µs of ~205 µs at M56; fusion can only touch
  front, tail and gaps (~80 µs). Inside a HIP/CUDA graph the launches already run back-to-back with
  ~0 gap, so launch-count fusion bought nothing on its own.
- Hand experiments that lost: separate-launch split unpool (extra cross-rank round ~9 µs), fused
  one-launch tail (0.4-8.5% slower), FlyDSL experts (5-10% slower than production Triton),
  scatter inside GEMM1. A non-fusion win: aiter `gemm_a16w16` for o_proj / unpool at 17..32 rows.
- Deployable path (`MUSE_MEGA_EPBAL=0`): M16 1.14x, M32 1.08x, M56 1.08x, M64 1.07x.

## G. Muse attention-output boundary, TP8 BF16, gfx942 (one persistent HIP launch)

o-norm / gate -> o_proj -> TP8 all-reduce -> post-norm / gated residual / pre-FFN norm -> fp32
router GEMM. Target 64 tokens.

| R | step | score | note |
|---|---|---|---|
| 1 | one launch, tile-level RS/AG over IPC, epoch flags | 1.016 | launch-floor model predicted ~18.5 µs saved, delivered ~1 µs |
| 3 | router reduce spread over tickets, agent scope for rank-local flags | 1.095 | sync scope |
| 4 | W_o prefetch moved out of the join, expert-split router AG, row x col split of phase C | 1.165 | overlap |
| 5 | 16 B loads / write-through stores after the acquire, owner-carried row scalars | 1.207 | sync / codegen |
| 7 | unconditional fragment-major A loads (the compiler had put vmcnt(0) before every row block) | 1.275 | codegen: A MMA 9.4 -> 6.1 µs |
| 9 | router algebra: logits = r2 * (Ps + r1 * Po); Ps computed during the waits, Po per AG tile | 1.463 | fusion-only: work moved off the critical path |
| 12 | AG push peer rotation | 1.497 | AG drain 4.48 -> 3.28 µs |

Losses: phase B on helper CTAs -15%, pull-mode AG -30% (2.6x remote bytes plus remote polling),
default-off arm code left in the hot instance +1.4 µs (register allocation changed). 256 tokens stays on
a declared fallback. A later TP4 / gfx950 port of the same tail gave 1.06-1.17x at kernel level and
no measurable e2e gain at M56 once it ran inside the serving graph.

## H / I. Qwen3-235B MoE EP8 fp8, gfx942: same operator, two baselines

H, tuned aiter/mori baseline (8 launches, ASM fmoe):
- R1 Triton two-launch design: 0.315x; the Triton fp8 skinny core streamed weights at 13-19% of HBM.
- R2 HIP fp8 MFMA persistent core: 0.78x (1 WG/CU at 491 VGPR+AGPR).
- R4 LDS-staged A over wide N, LDS-staged combine push writing contiguous 256 B rows: 0.957x.
- R5 unified monotone-ticket pool: 0.980x. R7 LPT pool order: 1.008x.
- R9 declared T<=32 fallback to the baseline.
- R10 hierarchical once-per-destination arrival publish: 1.047x, the first resolved win.
- Final 1.054x at T=64, 1.138x at T=128. Caveat: p99 per call 1.66 ms vs 0.87 ms frozen; the claim is a median.

I, scratch Triton/mori baseline (10 launches):
- R1 whole-grid G1 -> G2 barrier with system-scope tickets: 0.837x.
- R2 per-expert `g1_done[e]`, agent scope: 1.015x, with an intermittent stale first-call read.
- R3 correctness fix: 1.075x. R4 atomic expert-major claim: 1.100x.
- R5 tile-contiguous W1/W2 and per-T buckets: 1.204x. R7 register/config tuning: 1.245x.
- R10/11 fold K1 into K2, W1 prefetch issued under the dispatch spin: 1.249-1.258x.
- Independent verify: 1.118x / 1.247x / 1.283x at T 32 / 64 / 128.

Lesson: against a scratch chain the same fusion work shows 1.25x; against tuned library kernels it
shows 1.05x. Most of the "fusion gain" over a scratch chain is what tuned kernels already had. Pick
the production path as the baseline.

## J. Workflow-level lessons from these runs
- 3 of the 5 runs above (E, F, I) shipped an empty patch with verified >1.16x candidates, because two final
  selections shared one `mega_selected` tree and the evidence manifest no longer matched. Run one
  selection at a time or give each its own tree.
- 12 directions were not enough for a 4-rung ladder (one launch -> parity -> readiness -> levers);
  plan ~15+ when the skeleton starts below 1.0x.
- Rounds that only add <1% sit below a ~2-3% resolution floor; stop micro-tuning when 2-3 rounds in
  a row land there and change the structure or the bound instead.
