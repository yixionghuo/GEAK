# 02 — Kernel-level levers inside a fused kernel

Levers were measured with paired interleaved CUDA-graph A/B, rank-max, on six operators (case
ledgers in `05`). Unmarked numbers are MegaMoE V2 EP8, gfx950 (MI355X, 8 XCD, 256 CU), FlyDSL:
"@64" = 64 tok/rank single persistent kernel (graph µs), "@8192" = 8192 tok/rank two-launch
persistent kernel (ms or %). Evidence from the other operators is marked by source:

- `{attn}` Muse Spark attention block, TP4 verify, gfx950, one persistent HIP launch (`05` §E)
- `{moe}` Muse Spark MoE block, TP4/EP4 verify, gfx950, Triton front + HIP tail (`05` §F)
- `{attn-out}` Muse attention-output boundary, TP8, gfx942, one persistent HIP launch (`05` §G)
- `{qwen}` Qwen3-235B MoE EP8 fp8, gfx942, against a tuned aiter/mori baseline or a scratch chain (`05` §H-§I)

A lever seen on several operators is more likely to transfer. Numbers are calibration for the
*shape* of an effect; re-measure on your operator. Each lever is tagged:

- **[fusion-only]** — not expressible across a kernel boundary; this is where 1+1>2 comes from.
- **[restore]** — recovers efficiency the fusion lost; required for parity.
- **[shared]** — also speeds the unfused kernel; credit it with the fusion-off control.

Portability tags (see `SKILL.md` §0.2): `[AMD]` AMD Instinct ISA/topology, `[gfx950]` lowering,
bits, limits or numbers read on gfx950, `[FlyDSL]` DSL-specific, `[multi-rank]` cross-GPU only.
Untagged items are expected to transfer as mechanisms; their numbers still need re-measuring.
All evidence comes from gfx950 (MI355X) and gfx942 (MI300X); `[AMD]` marks mechanisms of their
ISA and topology, not verified on other AMD parts.
---

## 0. Index: what only one kernel can do (the 1+1>2 levers)

Simply chaining the stages inside one launch (`drain_all(A); barrier; drain_all(B)`) buys at most
the launch gaps and usually loses (`SKILL.md` §1 item 8; 0.685x and 0.837x skeletons). The gain
comes from the levers below, which do not exist across a kernel boundary. Grouped by what the
single kernel makes possible; details under each id. Evidence is from gfx950 and gfx942 only.

Prerequisite (not a gain, but nothing below pays without it): every stage back at its standalone
speed inside the kernel: B1 claim order for operand reuse (4.50 -> ~4.05 ms @8192), E1 tuned configs
per bucket, E2 all waves on one tile, F codegen (`09` S1).

| mechanism | id | lever | precondition | measured effect |
|---|---|---|---|---|
| **1. Start the next stage's independent work early** | A1 | weights / metadata before the activation gather; wait for readiness inside the tile | consumer has operands that do not depend on the producer | -1.3, -0.7, -0.4 µs @64 |
| | A3 | claim the next unit near the end of the current one and issue its first loads | exact-size buffer descriptors; claim at K-step 24/28, not at unit start | -1.2 µs @64 |
| | A6 | build the consumer's pointers, descriptors, setup during the readiness wait | a wait exists on the path | -1.7 µs @64; `{attn}` loads at entry ~4 µs/layer |
| | G9 | arrive before a large weight stream, wait after it | the stream would otherwise share the in-order vmcnt with latency-critical loads | `{attn}` 164.6 -> 157.1 µs |
| **2. Move work to where there is slack** | A10 | algebraic split: compute the part of a later stage that needs only early data during the waits | exact rewrite; re-check the error budget | `{attn-out}` router 1.30 -> 1.46x (+18 pt) |
| | A11 | do a split-K / split reduction inside a communication round that runs anyway | a reduction next to an existing exchange | `{attn}` 1.081 -> 1.118x (with G12) |
| | G8 | put merge / reduction units on CTAs that have no item in the current region | idle CTAs exist; the region is not HBM bound (else L34) | `{attn}` 1.064 -> 1.081x (with G7) |
| | B7 | while waiting for a payload, peek and CAS-claim a ready consumer unit | non-destructive peek; never block while holding a producer claim | -0.7% uniform, -0.9% skew @8192 |
| | B6 | reshape the tail: split the last tiles, size reducers to the tail | tile-size limits (BN/2 only for BN=256) | -0.8 µs, reducers -1.5 µs @64 |
| **3. Hand over between stages more cheaply than a kernel boundary** | C9 | readiness counters per region / request instead of grid barriers | monotone counters, whole grid resident (E5) | each barrier 4-6 µs at 1 CTA/CU `{attn}` |
| | C2 | same-XCD handoff: no release fence, L1-only invalidate on the consumer | producer and consumer units on one XCD (B4) | -6 µs @64 |
| | C3 | write-through stores + waitcnt instead of a release fence | data read by another die or rank | per-block fence cost +0.41 ms @8192 avoided |
| | C5 | drop launch handshakes, clear counters before the gate, one plan word | every launch ends with a cross-rank exchange | -2.5, -7, -1.5, -2.5 µs @64 |
| | C6 | publish once per workgroup after the loop, not per unit | a coarse barrier before the reduction exists anyway | -0.8% @8192 |
| | G11 | the last-arriving CTA finishes a split reduction | partials visible to it; monotone counter; merge fits one CTA | method; `{attn-out}` ticket form 1.016 -> 1.095x |
| **4. Pipeline stages across CTAs instead of across the grid** | B9 | each CTA drains its producer share, then takes ready consumer work; no global stage barrier | multi-wave or unbalanced producer (little window on balanced routes, L27) | measured through A9 and B1 |
| | A9 | reduction / combine as a third queue in the same grid | enter only when the CTA's own queues are empty | 4.62 -> 4.50 ms @8192 |
| | B3 | sharded claim heads, contiguous claims | many claims per launch | one unsharded head cost more than the GEMM2 math |
| | B4 | keep a producer's tiles and their consumer units on one XCD | multi-XCD part | enables C2; stealing across XCDs +1.3..8 µs |
| | B5 | skew-gated reorder of consumer units across peers | skew statistic at the granularity of the imbalance | skew 0.58x -> 1.21x |
| | B10 | send once per destination; send route counts early | multi-rank dispatch | -1.0, -1.4 µs @64; `{qwen}` front 33.9 -> 23.4 µs |
| **5. Share on-chip resources across stages** | E3 | LDS = max of the role lifetimes, not the sum | roles do not hold LDS at the same time | fits 160400 / 163840 B @8192 |

Not covered yet (no measured case): handing a producer's output to its consumer in registers or
LDS inside one CTA without a global-memory round trip (the flash-attention pattern across two
GEMMs), and role-specialized waves with different resource shapes (only the generic A8).

---

## A. Overlap and prefetch (fill the idle pipe with dependency-free work)

The dependency is on the consumer's **first read** of the producer's output, not on the consumer
task. Everything before that read — descriptor setup, scheduler/barrier state, weight and scale
loads, index preprocessing — is dependency-free and can run during the producer. A kernel boundary
makes it inexpressible.

A1. **Weights before activations** [fusion-only/shared]. Each tile issues its B (weights) before
the A gather; a dependent GEMM unit issues its B and route metadata first, then waits for the
producer's readiness *inside* the tile, then loads A with a row mask.
@64: B before A -1.3 µs; ready-wait inside tile after B issue -0.7 µs; A-scale issued first -0.4 µs.

A2. **Lookahead depth 2 in the K loop** [shared] [AMD: `vmcnt` note]. Keep 2 K-steps of B in flight.
@64: 1->2 -0.7 µs; 3 or 4 +1..+4 µs (register/issue cost). @8192 the G2 K-loop waited
`vmcnt(2)` before every BK256 back edge; a two-deep W2 prefetch is the open lever there.
`{attn}` a deeper per-CU K/V load pipeline with warm page ids: 1.118 -> 1.156x. `{qwen}` weight-ring
depth was neutral and a continuous cross-item ring lost 3-6 µs: depth pays only while registers last.

A3. **Cross-tile prefetch** [fusion-only] [AMD: buffer-descriptor drop]. Claim the next unit before the current one ends (K-step
24 of 28) and let the last two K-steps issue the next unit's first B. @64 -1.2 µs. Claiming at
K-step 20 was slower; claim-ahead at unit *start* (no extra bytes in flight) +1..4 µs.
Needs exact-size buffer descriptors so out-of-range prefetch past the last tile is dropped by HW.
`{qwen}` a next-unit L2 weight prefetch at unit boundaries lost (0.945-0.993x, worse with depth):
in-order vmcnt plus XCD L2 overflow. Cross-tile prefetch pays only as issue of the next unit's
first loads, not as a separate warm-up.

A4. **Half-tile ring for the dependent GEMM's B** [shared]. GEMM2 B half-tile ring of 3: -0.4 µs
@64. Deeper whole-tile buffering +2-3 µs and VGPR spills.

A5. **Issue order relative to waits** [shared] [AMD]. An in-order `s_waitcnt vmcnt(n)` also drains a
prefetch issued after it. Issue B and A-scales *before* the barrier and before the next A DMA:
~+3% @8192 on a build that lacked it. `{attn}` the Wo slice prefetch into LDS during attention
had to be issued so it does not sit in front of attention loads in the vmcnt queue; `{attn-out}`
a per-row-block RS push pipeline turned negative once the loads overlapped, for the same reason.

A6. **Hoist consumer setup before the plan/readiness wait** [fusion-only]. Peer tables, pointers,
descriptors of the dependent stage built while waiting for the plan: -1.7 µs @64.
`{attn}` all weight loads issued at kernel entry plus epilogue operands prefetched: ~4 µs per
layer (with F12, 1.156 -> 1.174x). `{attn-out}` W_o prefetch moved out of the join.

A7. **Overlap only through bytes that must be read anyway** (weight-streaming regime). Any *extra*
early HBM read — warm-up, preload during the dispatch front end, MALL prefetch — cost +1..+10 µs in
every variant: it competes with the payload writes. Prefetch = reorder required reads, never add reads.
Confirmed on three more operators: `{moe}` w13 prefetch during the front cost 0.31 µs/MB and saved
0.12 µs/MB; `{qwen}` first-item weight touch +10 µs, MALL prefetch 2-6 µs per 100 MB against an 8 µs
go bar; `{attn}` moving the gate GEMM's weight stream into idle CTAs during attention made S56 16%
slower. The form that paid: `{qwen}` W1 prefetch issued *under a wait that exists anyway* (the
dispatch-flag spin), +0.3-1.4%.

A8. **Split loader** (generic recipe). Give activation and weight operands different loader waves:
the activation loader waits on the producer's readiness, the weight loader runs ahead up to
`num_stages`; both meet at the per-stage full barrier the MMA loop already uses. Correctness
argument unchanged. Costs registers/LDS per stage — usually the right way to spend them (raising
occupancy instead can buy out the prefetch).

A9. **Combine/reduction as an overlapped third queue** [fusion-only]. Stage1+Stage2 persistent
with a separate combine launch 4.62 ms; combine as an irreversible third queue in the same grid
4.50 ms @8192. A per-row fine interleave of GEMM2 and combine was a net loss and deadlocked twice;
the third queue (enter only when own G1/G2 queues are empty, per-token lane-0 wait) is the form
that pays.

A10. **Compute early what depends only on early data** [fusion-only]. Look for terms of a later
stage that need only data available before a wait. `{attn-out}` the router after o_proj and two
norms was rewritten as `logits = r2 * (Ps + r1 * Po)`: `Ps` (residual part) computed during the
cross-rank waits, `Po` per gathered all-gather tile, the epilogue only finalizes: 1.30 -> 1.46x,
the largest single step of that run.

A11. **Split comm-carried reductions into compute** [fusion-only] [multi-rank]. `{attn}` o_proj
split-K x2 with the second K half summed inside the reduce-scatter that runs anyway; with a
single-round sliding merge: 1.081 -> 1.118x.

**Diagnose before choosing A-levers:** if all pipes are low, `SQ_WAIT_ANY` high and inter-kernel
gaps ~0, the stage is latency/dependency stalled — more independent work *per wave* (A-levers) helps,
more waves (occupancy) does not. If unrolling/deeper outstanding work does not help (U=4 4.4535 vs
U=1 4.4476 ms), the residual is not latency-bound; stop there. A region already at its byte floor
(`09` §3) gains nothing from any A-lever. Start from the symptom table in `09` §4, which maps what
you measured to the levers of this file and to the `04` rows not to retry.

---

## B. Scheduling, claim order and locality (restore and exceed the original grid's reuse)

B1. **Claim order is a tuning surface** [restore]. The persistent scheduler replaces the original
grid's block->tile map; a linear claim loses the operand reuse that map gave. Column-grouped order
for the dependent GEMM — walk a group of GM m-blocks down one n-block column so concurrent claims
share one weight slice in L2:
```
group   = pos // (GM * n_blocks)
r       = pos - group*GM*n_blocks
gm_eff  = clamp(total_m - group*GM, 1, GM)
n_block = r // gm_eff
m_block = group*GM + (r - n_block*gm_eff)
unit    = m_block*n_blocks + n_block
```
@8192, C=1: GM=1 4.681, 8 4.419, 16 4.089, 32 4.047, 64 4.023 ms (GM 48-128 flat). G2 span 2.05 ->
1.36 ms while G1/G2 overlap stayed ~3% — the gain is G2 *efficiency*, not overlap. This single lever
was worth more than all removed waiting. Keep the weight load in L2 (an `sc1` L2-bypass on W2 was
+5.5%). Keep the mapping a bijection over all units, or a CTA waits on a unit nobody runs.

B2. **Chunk size x order, swept together** [restore]. With column grouping: C=1 4.089, C=2 4.202,
C=4 4.424, C=16 4.880 ms (tail quantization over ~11.6k units on 256 CTAs). C=16 only belonged to
the m-major order. For a drain smaller than ~64 units per shard keep m-major.

B3. **Sharded claim heads, contiguous claims** [fusion-only]. One unsharded head serialized ~11k
agent atomics on one cache line and cost more than the GEMM2 math. Shard heads, one 64 B line each
(`shard = ticket & (S-1); unit = shard + local*S` for the producer stage). But strided sharding of
the *consumer* chunks (s, s+4, s+8...) broke m-block locality and was +3.7% slower; keep consumer
claims contiguous in position space.

B4. **Locality-preserving split across dies** [fusion-only] [AMD]. Give each XCD a contiguous m-major
slice (all n-tiles of an m-tile on one XCD, so A stays in that XCD's L2) and put the dependent
units after the producer units *on the same XCD counter*. Then the producer->consumer handoff needs
no release fence and only an L1 invalidate (see C2). Work stealing across XCDs re-reads A from HBM
and needs L2 writeback/acquire: +1.3..+8 µs every time it was tried @64.

B5. **Skew-gated reorder** [fusion-only] [multi-rank]. On a rank-mixed-skew route one expert holds most rows,
laid out by source rank; consecutive m-blocks return to one peer and the column order serializes
P2P stores on one link (skew 0.58x). Gate on `max_expert_tiles*4 > num_m_tiles` and permute
`m' = (m % npes)*(total_m // npes) + m // npes` before grouping: skew 1.21x, balanced order unchanged.
Use a statistic at the granularity the imbalance lives at (per expert, not per rank).

B6. **Tail shaping** [fusion-only]. Split each XCD's last m-tile into BN/2 units (-0.8 µs @64; only
for BN=256 — BN/2 of a 128 bucket = 64 tile gave wrong results; splitting the last 2 m-tiles was
worse). Size reducer/consumer counts to the tail (combine reducers 64 -> 256: -1.5 µs @64).

B7. **Non-blocking fill while waiting** [fusion-only]. A CTA holding a producer stripe whose payload
is not ready peeks the consumer head, checks that position's readiness, claims exactly it with CAS,
runs it, returns: -0.7% uniform, -0.9% skew @8192. An atomic-add claim instead tripled fills for
-0.2%. Never take a *blocking* consumer wait while holding an unexecuted producer claim.

B8. **Peek, don't speculate** [fusion-only]. Preempting with consumer work must use a
non-destructive peek (atomic add 0), claim only if ready, fall through to producer work on a miss,
and block on the consumer's readiness only once the producer queue is drained. A speculative
destructive claim of unready consumer work starved the producer and livelocked. Gate preemption to a
fraction of CTAs (`ticket % 6 == 0`) and to a skew predicate (5/4 of balanced share); on a uniform
route it costs +0.30 ms, on skew it saves 0.51 ms.

B9. **Phase-staggered CTA pipeline, not grid-wide sequence** [fusion-only]. Each CTA drains its own
producer shard, then switches to ready consumer work; later to reduction. No global "all producer
done" barrier between queues. Benefit = tail filling and overlap of downstream VMEM/P2P/epilogue,
not simultaneous MFMA. Reject pseudocode of the form `drain_all(A); barrier; drain_all(B)`.
On a balanced route most producer tiles finish together, so per-m-tile readiness opens little
window; the measured uniform gain came from B1/B3 machinery and the third queue, not from fine
GEMM1||GEMM2 overlap. Do not build the fine-ready frontier for a uniform route.

B10. **Dedup and early metadata** [fusion-only] [multi-rank]. Send each token's payload once per destination rank
(not per top-k slot): -1.0 µs @64. Send per-expert route counts early (epoch-tagged, parity
double-buffered) so the destination writes metadata before payload flags arrive: -1.4 µs @64.
`{qwen}` publish arrival once per destination rank instead of once per item: front end
33.9 -> 23.4 µs, the first resolved win of that run (1.012 -> 1.047x).

B11. **Balance ragged work across CTAs first** [shared]. Work of uneven size per request (KV
length) or per expert must be split into equal pieces before any fusion lever. `{attn}` equal KV
tokens per CTA across requests: F56 attention 116 -> ~96 µs, 1.22x on the full layer with no fusion
at all, the largest lever of that run. `{qwen}` a unified monotone-ticket pool with half-width items:
G1 max/mean 123.8/70.7 -> 92.5/73.6 µs; LPT order then -6 µs @T=64.

B12. **Balance across ranks** [multi-rank]. The slowest rank sets the span. `{moe}` dynamic EP
balance removed a 37-vs-28 active-expert skew (+6.4%, the largest MoE lever), but with all 256
experts on every rank it costs +5.1 GB per rank per layer; the deployable form is partial
redundancy (EPLB-style hot subset).

B13. **Split replicated work across ranks** [multi-rank]. `{moe}` each rank computes 1/4 of the
pooler / unpool columns and exchanges slabs instead of every rank computing all of them. Pays only
when the exchange rides a cross-rank round that already exists; as a separate launch it lost
(+9 µs for the extra round).

B14. **Spread cross-rank traffic over peers, push not pull** [multi-rank]. `{attn-out}` rotate
all-gather destinations per unit and row: AG drain 4.48 -> 3.28 µs. Pull-mode AG: -30% (2.6x
remote bytes plus remote polling). Helper CTAs for a phase gated by remote arrival: -15%.

---

## C. Synchronization scope and memory order (the cost that eats the fusion gain)

C1. **Narrowest correct scope.** [AMD] [gfx950: lowering] Rank-local edges: agent scope. Peer edges: system scope on
peer-visible storage. On gfx950 `fence_agent_release` = `buffer_wbl2 sc1` (cross-XCD L2 writeback);
system acquire = `buffer_inv sc0 sc1`. Per-tile/per-CTA fences: per-CTA system release +54 µs @64;
per-tile agent release + XCD-interleaved GEMM1 left the single kernel 75-100 µs slower.
`{attn-out}` router reduce on tickets with agent-scope rank-local flags: 1.016 -> 1.095x. `{qwen}`
a whole-grid G1 -> G2 barrier with system-scope tickets: 0.837x (~71 µs of handshakes).

C2. **Fence-free same-XCD handoff** [fusion-only] [AMD]. Producer: stores, `s_waitcnt(0)` + workgroup
barrier, thread 0 agent atomic count; the completing count raises the ready word to the launch
epoch. No release fence. Consumer: poll ready word (`>=`, never `==`), invalidate **L1 only**
(-6 µs @64). Valid only if B4 keeps both on one XCD.

C3. **Write-through instead of release** [fusion-only] [gfx950: cache bits]. Mark payload/scale stores write-through
(`sc0|sc1` = 17; P2P with `nt` = 19), `s_waitcnt vmcnt(0)`, then the completion atomic. A per-block
`fence_system_release` measured +0.411 ms (~200 blocks). Note: write-through on *GEMM1 output* in
the single-XCD-local recipe was +10 µs @64 — use it where cross-die/peer visibility is needed.

C4. **One release per launch, not per CTA.** [multi-rank] [gfx950: cache bits] Exit tickets after `s_waitcnt(0)` + barrier with no
fence; only the last ticket holder does one system release before signalling peers; reducers
invalidate L1 after seeing all peer flags. Combine loads with cache 19 need no acquire fence;
output stores rank-local cache 2: -0.5% @8192.

C5. **Remove handshakes** [fusion-only] [multi-rank]. No launch handshake when every launch ends with the
cross-rank combine exchange (-2.5 µs @64); owner clears work counters with several lanes and one
release *before* opening the gate so the first claim overlaps the plan wait (-7 µs); plan as one
i64 word `(epoch<<32)|num_valid`, one CTA per XCD acquires and forwards it (-1.5 µs); gate read
without acquire, epoch with sc0|sc1 (-2.5 µs).

C6. **Deferred publication** [fusion-only]. With a coarse barrier before the reduction, per-unit
drain+barrier+add is unnecessary: count units in LDS, publish once per workgroup on leaving the loop
(-0.8% @8192). Publish once per tile *after* the hot loop, never inside it (C7 / F4).

C7. **Correct publish shape.** [gfx950] [FlyDSL] `fx.barrier()` = `s_waitcnt lgkmcnt(0); s_barrier` — it does **not**
drain `vmcnt`. Publish = every wave waits its own stores (or releases), barrier, then one thread
atomic. Thread-0 `vmcnt(0)` alone publishes a partially written tile (passes replay 1, fails replay
2). Publish at the head of the next iteration rather than the loop tail if the tail form faults.

C8. **Counters.** One monotone counter per item on its own 64 B line, never cleared; waiters target
`generation x expected`; pad omitted items to the target before the epoch gate; a clear, if
unavoidable, happens in the ticket-0 block before the gate. Ready counters one per 64 B line:
-0.8 µs @64. `{moe}` one 128 B line per flag, together with C10: scatter_done 24.6 -> 19.8 µs.

C9. **Readiness edges, never grid barriers.** Every fused run that started with phase barriers
started below 1.0x: `{attn}` 0.685x with grid barriers, 4-6 µs per barrier at 1 CTA/CU;
`{qwen}` 0.837x. Monotone per-region / per-request readiness counters replace them (`{attn}` used
per-request `att_ready` counters; a last-arriver merge, G11, is the other form).

C10. **Peer stores without per-store drains** [multi-rank] [AMD]. `{moe}` plain global
write-through stores to peer memory each got `s_waitcnt vmcnt(0)` (one xGMI round trip per store) in
Triton; a pointer with `ptr_range` lowered them to buffer stores. `{attn-out}` 16 B loads after the
acquire and 16 B write-through pushes: +3.5 pt.

C11. **Data as its own flag** [multi-rank]. A sentinel in the payload (BF16 -0.0; senders write
+0.0 for -0.0) marks arrival: no fence, no separate flag, reset-free and graph-replay safe. The
fastest IPC protocol measured on the Muse kernels `{attn}` `{moe}`.

---

## D. Queue structure (one state machine per CTA) [persistent]

Decision order that measured best at 8192 (one block-uniform outer loop):
```
if carried consumer chunk has continuation: run next unit          # 0 atomics, 0 polls
elif CTA in skew cohort and peeked consumer chunk is ready: claim + run
elif producer stripe remains: claim, wait payload, run, publish once
elif consumer chunk remains: claim, wait last in-range dependency, run
else: enter reduction queue permanently (claim block, per-wave item, lane-0 wait)
```
Hot-path budget per unit: continuation 0 atomics/0 polls; new consumer chunk 1 claim + 1 wait;
producer stripe 1 claim + 1 system increment; reduction 1 claim per NW items, 1 lane-0 poll per
item, no fence. Broadcast only `kind` and `unit` through LDS; carry continuation as block-uniform
scalars of the outer loop. Loop-top barrier mandatory (separates previous unit's LDS reads from the
next scratch write).

---

## E. Configuration and resource shape

E1. **Per-bucket selector** [restore]. Wire the baseline's tuned per-shape configs into the fused
path. Tuned constants are derived per bucket in the selector and clamped by budget (LDS, residency);
only structural on/off is an env switch. Every config value belongs in the JIT key.
`{qwen}` tile-contiguous W1/W2 layout plus per-T buckets: +9.4%, the largest step of the scratch
run; re-tuning registers and config for the fused body: another +3.5%.

E2. **Use all waves on one tile** [restore] [gfx950: numbers]. Native NW=8 slab with BN widened by `waves/4` (8192:
BM64/BN512/BK256/NW8) instead of two lockstep 4-wave halves: halves +0.53 ms. Assert emitter limits
(BN <= 512 per unit; wider silently drops columns).

E3. **LDS by lifetime** [fusion-only] [gfx950: 163840 B limit]. `group_segment = max(producer_pool, consumer_slab) +
shared_scale_storage` (8192: max(131072, 131728) + 28672 = 160400 <= 163840). Additive allocation
overflows. Every helper address is relative to the caller's slab base.

E4. **Name the binding resource before chasing occupancy.** [AMD] [gfx950: VGPR units] At 256 VGPR a 512-thread block is
1 WG/CU regardless of LDS; cutting LDS to 2 WG/CU cost +2.33 ms. VGPR budget = arch VGPR + AGPR
(256/lane at waves_per_eu=2). Profiler VGPR on gfx950 is in units of 2.
Register walls on the other operators: `{attn}` 2 waves/SIMD would need 415 registers per wave;
`{qwen}` the HIP expert core ran 1 WG/CU at 491 VGPR+AGPR, and 2 waves/SIMD or 2 CTAs/CU was slower in
every configuration. Design the fused body for 1 wave/SIMD unless the register budget is shown.

E5. **Residency is correctness.** Persistent grid = exactly the co-resident count derived from
emitted metadata, asserted on the host; every participant index inside `[0, resident)`. One added
atomic or live value can cross a limit and turn every inter-CTA wait into a deadlock.
`{moe}` a dependent G1+G2 grid ordered by program_id deadlocked: program_id order does not
guarantee that the producers are resident. Claim work by ticket.

E6. **One resource shape for every role.** Price the op with the worst fit; gate the fused path to
the bucket where it pays and keep the scattered path elsewhere (legitimate design).
`{qwen}` T<=32 and `{attn-out}` M>128 call the unmodified baseline inside the same op, declared in
the path marker. Production-tuned stages you would have to rewrite in another DSL to put inside
the kernel usually belong outside it: `{moe}` FlyDSL experts were 5-10% slower than production
Triton; `{qwen}` a Triton fp8 skinny core streamed weights at 13-19% of HBM.

E7. **Fixed costs do not shrink with the problem.** Measure handshakes/fences/launches at the small
guards; don't compute padding (tile far larger than real rows multiplies work).

---

## F. Codegen inside the fused body (read emitted metadata first)

F1. **LICM spill** [shared]. A compile-time-constant ping/pong offset in a peeled last K-step was
hoisted and spilled (32 scratch reloads per tile). Anchor it to the per-tile runtime row base:
spills 53->20, private 216->84 B, ~+1% @8192. Generic form: pass the value through an identity
inline asm taking a per-iteration runtime operand. Verify in ISA; re-check after toolchain bumps.

F2. **Uniform values in SGPR** [shared] [AMD]. A workgroup-uniform A-tile start row in a VGPR cost 4
`readfirstlane` waterfall loops per K iteration; load to SGPR: G1 K-loop -57 µs per WG, ~+0.85%
@8192 (per-bucket: one build -1.4% at bs128, another no change).

F3. **Vectorized epilogue** [shared]. SiluQuant epilogue 8 columns per lane (one 8 B FP8 store,
2 amax shuffles per 8 columns): 245 -> 164 µs per WG, ~+1.5%. F1+F2+F3 stacked: 4.115-4.127 ->
3.925-3.933 ms (-4.6%), spills 75->31 — more than the sum measured separately. Keep behind the
large-bucket compile key so other buckets stay byte-identical.

F4. **Atomics out of the hot loop.** An atomic RMW in a GEMM loop tail shifted register allocation
and faulted at an unrelated address. Publish once per tile after the loop / at next-iteration head.
Tell: replace the publish with a benign constant store; if it faults identically, the bug is a
latent address/regalloc defect elsewhere (fix address init, pin bases to SGPR, bound buffer
resources to real extents).

F5. **No nested loop around a GEMM body** — carry the continuation in the outer loop (nested ~2.6x).
Build descriptors inside the unit frame; hoisting to the outer frame extends live ranges into the
reduction and spills SGPRs. Never coarsen the work unit to escape a frame fault; shrink live ranges
(per-unit `@jit` helper).

F6. **Keep 16 B vector copies.** [FlyDSL] Fix scalarization errors by matching the baseline operand form
(wave-uniform resource, dtype, offset type), never by splitting stores.

F7. **FlyDSL specifics.** [FlyDSL] Runtime control flow only in kernel body / `@flyc.jit` helpers; masks and
`select` choose values, never who performs a side effect (an all-lane masked atomic multiplies
atomics 64-512x); loop-carried vars must be pre-initialized typed zeros on every path (not under a
`const_expr` branch, not None); barrier between two thread-0 regions including across the back edge.

F8. **No `const`/`__restrict__` on buffers written by other blocks of the same kernel** — the
compiler hoists the load above the readiness spin. Verify order in ISA; test with poisoned outputs.

F9. **Fix shared-state races on every path.** [FlyDSL: the MegaMoE case] The baseline persistent loop broadcast `work` through
LDS word 0 then overwrote it with `has_work` without a barrier (relL2 0.058-0.075 nondeterministic).
Own LDS slot + each wave computes `has_work`: deterministic (0.0532). A baseline that is not
bit-exact on repeated input has a bug; find it before tolerating it.

F10. **Unconditional operand loads** [AMD]. `{attn-out}` a conditional load made the compiler put
`vmcnt(0)` before every row block; loading unconditionally (masking the use) and fragment-major cut
the A MMA 9.4 -> 6.1 µs (1.207 -> 1.275x).

F11. **Compile out default-off arms** [shared]. `{attn-out}` an arm left in the hot instance behind a
runtime flag changed register allocation: +1.4 µs. Experimental arms are compile-time switches.

F12. **Widest MFMA the arch offers** [gfx950]. `{attn}` one `v_mfma_f32_16x16x128_f8f6f4` instead of
four 16x16x32 FP8 MFMAs, with a lane-parallel segment table (part of 1.156 -> 1.174x).

F13. **Contiguous rows for peer writes** [multi-rank]. `{qwen}` stage the combine output in LDS and
push 256 B contiguous rows instead of 16 B per-lane fragments: 0.869 -> 0.955x.

---

## G. Attention, epilogues and small GEMMs inside a fused block

Measured on the Muse Spark attention block `{attn}` (QKV GEMM -> QK norm / RoPE / KV write ->
paged verify attention -> merge / norm / gate -> o_proj -> RS/AG -> boundary norm, one persistent
HIP kernel, 1 CTA/CU, gfx950). Diagnosis that led to each item: `09` §6.

G1. **One wave per KV head over full rows** [shared]. Four waves walking the same
(request, KV head) segment fetched each 256 B K/V row once per 64 B head slice (2x over-fetch).
Wave = KV head, full rows: F56 attention 153 -> 122 µs.

G2. **Piece size against partial bytes** [shared]. Token-balanced pieces need a minimum length:
at ~2 steps per CTA the 64 KB fp32 partial of a sliding piece was as large as the K/V it read.
A minimum of 4 32-token steps saved ~5 µs at S56; for small windows prefer fewer, longer pieces.
Per-request piece splits helped sliding layers and hurt the full layer (141 vs 125 µs).

G3. **Static softmax when scores are bounded** [shared]. With RMS-normed q and k the score is
bounded, so `p = exp2(s * scale)` needs no running max, vote or rescale: 177 -> 161 µs.
Precondition — numeric headroom, checked on the host for every call shape, not assumed:
the largest unnormalized term is `2^max_log2_score`, and the denominator sums up to `max_keys` of
them, each multiplied by V (FP8 e4m3 max 448) in the PV product. Keep the fp32 sum below 2^126:
`max_log2_score < 126 - log2(max_keys) - log2(448)` for FP8 V (use the max |V| of your format).
`max_log2_score` follows from the norm bound: |q|·|k| <= (RMS bound)^2 · head_dim, times
`sm_scale · log2(e)` and any q/k quant scales. If the bound fails for some shape (long context,
large scales), that shape keeps the running-max path. Any softmax without a running max (attention,
router, sampling) needs the same check; reject it if the input has no bound (`08` §2).

G4. **Take VALU out of the step body** [shared]. At 1 wave/SIMD the step loop was VALU bound: fold the
softmax scale into Q, compute the denominator with an MFMA, keep the masked path cheap. A branchless
step body (+~128 VALU per step) cost +50 µs.

G5. **Small index tables in LDS, kept as vector values** [AMD]. Stage the page-id table (or any
per-item index table: block tables, expert offsets, segment starts) per piece in LDS and warm it
before the item starts, so the hot loop reads indices with LDS latency. Do not make a VMEM-loaded
index uniform with `v_readfirstlane`: it forces a `vmcnt` wait that drains every load issued before
it. `{attn}` page id via VMEM + readfirstlane: +50 µs at F56; page ids in LDS were part of the
125.7 -> 114.5 µs step.

G6. **Unconditional, clamped reloads in a prefetch ring** [AMD]. `if (step < n) reload` made the slot
registers a loop-carried phi whose back-edge copy forced `vmcnt(0)` every NPF steps; reload always
with the index clamped: 125.7 -> 114.5 µs.

G7. **Low-precision partials** [shared]. Partial O in bf16 with fp32 max and denominator halved the
partial bytes (sliding item work 8.8/12.8 -> 7.7/9.7 µs mean/max); relL2 unchanged. Budget the
error margin first: FP8 QKV had already taken the output to ~10x the baseline relL2 at small M.

G8. **Merge off the critical path** [fusion-only]. Put merge units on the CTAs that hold no
attention item (highest ids under a per-request split; -2.5% if not). Measured on `{attn}`: merge
11 µs vs 3.7 µs for the baseline merge kernel before this change; merge on idle CTAs plus bf16
partials (G7): 1.064 -> 1.081x.

G9. **Split arrive and wait around a weight stream** [fusion-only]. The 128 KB/CTA Wo prefetch gave
no gain in any single position because it shared the in-order vmcnt with the merge; arriving before
it and waiting after, with the last arrivers deferring Wo past their publish, paid: 164.6 -> 157.1 µs.

G10. **Issue all loads before math in small stages** [shared]. QKV GEMM re-tiled into 256 equal units
(144 columns x K/8) with every K step in flight: 29 -> 12 µs. QK epilogue with all loads issued
first and packed stores: 8.4 -> 4.0 µs. Weight loads issued at kernel entry with a lane-parallel
segment table instead of a serial thread-0 one moved the head of the critical path by ~1.5 µs.

G11. **The last-arriving CTA does the merge** [fusion-only]. Generic form for any split reduction
(split-K GEMM, split-KV attention, partial norms): every producer CTA writes its partial, releases,
and does one atomic add on a per-output counter; the CTA whose add returns `expected - 1` reads
all partials and finishes the output. No barrier, no separate merge launch, no CTA waits for
others; the merge starts the moment the last partial lands. Requirements: the counter is monotone
(`generation x expected`, `02` C8) or reset by the last arriver itself; partials are visible to the
last arriver (release by producers, acquire by it); the merge work fits one CTA or is split across
the last arrivers of sub-groups. Method, not measured on the Muse kernels: `{attn}` planned a
last-arriver merge per (request, KV head) but never built it (estimated L2-bound at 2-7 µs vs 5.4 µs
for the distributed merge); its 1.064 -> 1.081x came from G8 (merge units on idle CTAs) and G7 (bf16
partials). A related, measured form on `{attn-out}`: the router reduce spread over tickets with
agent-scope flags, 1.016 -> 1.095x.

G12. **One merge round when the pieces fit** [shared]. If the number of pieces per output is small
enough for one CTA (a sliding-window layer whose window spans a few pieces; a split-K of 2-4), merge
them in a single round instead of a tree or a capped batch: each extra round is a dependent step
(publish + poll + reload). `{attn}` merging all 17 pieces of a group in one round beat a 16-piece
cap with a second round; the single-round sliding merge together with the split-K in the RS
(A11) took 1.081 -> 1.118x. Large fan-in (full-attention layers with long context) still needs the
G11 hierarchy; choose per layer type.

---

## N. Norm and epilogue fusion (row-wise work after a GEMM or a communication step)

Evidence here is thinner than for GEMM/MoE: the `{attn}` QK epilogue and boundary norm, the
`{attn-out}` post-norm / gated residual / pre-FFN norm / router tail, and MegaMoE's SiluQuant
epilogue. Items without a measured number are marked as method.

N1. **A norm needs the whole row: choose who owns it.** RMSNorm/LayerNorm after a GEMM or an
all-reduce reduces over the full hidden dimension, so a tile that holds part of a row cannot finish
it (`01` §2 condition 1). Options, in order of preference: (a) the epilogue tile spans the whole row
(N tile = hidden), possible when hidden is small or M is tiny; (b) the row's owner CTA carries the
row statistic: producers publish partial sums of squares per row, the last arriver (G11) or the
owner computes the scale; (c) split rows x columns so each CTA owns full rows of a column block
after the communication. `{attn-out}`: a row x column split of that kernel's phase C, together with
W_o prefetch out of the join and an expert-split router all-gather, 1.095 -> 1.165x; owner-carried
row scalars with 16 B loads / write-through stores, 1.165 -> 1.207x (levers not isolated).

N2. **Chain the row-wise ops in one pass.** Residual add, gated residual, the next norm, the quant
of the normalized row and the next layer's router/projection input can be produced from one read of
the row while it is in registers; every extra pass is a full read of the activation. `{attn-out}`
post-norm, gated residual and pre-FFN norm run inside the same fused kernel after the all-reduce
(part of the 1.016 -> 1.207x path; not isolated). Method: count row reads in the fused kernel against the
number of row-wise ops; more than one read per row is the target.

N3. **Move scales and constant factors into neighbours** [shared]. Fold the softmax scale into Q
(G4), norm weights into the next GEMM's weights where the math allows, per-row scales into the
epilogue store; each removes a VALU pass at no byte cost. Algebraic rewrites can move whole terms
off the critical path: the router after two norms became `logits = r2 * (Ps + r1 * Po)` with `Ps`
computed during the cross-rank waits (A10, +18 pt). Check the numerical error budget after each
fold (G7 note).

N4. **Quant in the producer's epilogue, vectorized.** The activation cast belongs in the epilogue of
the stage that produced the activation, not at the head of the consumer (pulling it into a
persistent consumer as a separate phase cost +16.5..+65 µs, L18). In the epilogue: per-1x32 amax
with lane shuffles, several columns per lane and one wide store (MegaMoE SiluQuant 8 columns per
lane: 245 -> 164 µs per WG, ~+1.5% e2e, F3).

N5. **Small epilogues: all loads first, packed stores** [shared]. QK norm / RoPE / KV write with all
loads issued before any math and packed stores: 8.4 -> 4.0 µs `{attn}` (G10). In short epilogues the
cost is the dependent load latency, not the math; one round trip instead of several is the lever.

N6. **Keep the epilogue's extra outputs cheap** (method). Fused epilogues often write several outputs
(normalized row, residual stream, FP8 copy and its scale, KV cache). Write each once, contiguous,
with the cache policy its consumer needs (write-through only where another die or rank reads it,
C3), and test every output in the UT (`08` §5).

