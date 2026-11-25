# 01 — Decide what to fuse, and in what order to fuse and optimize

## 1. What fusion can buy, and what it cannot

A fused/persistent form gives exactly one thing a multi-launch form cannot: a consumer may start on
*part* of a producer's output before the producer has finished, and work on the far side of a kernel
boundary becomes *expressible* during the producer (prefetch, prologue, descriptor setup). Fewer
launches, warm caches and no weight re-load are available more cheaply by other means.

Fusion removes **waiting** (launch gaps, partial-wave tails, grid-wide joins, cross-rank barriers).
It never removes **work**. The cost of the work moves inside the fused kernel in either direction,
decided by claim order, chunking, resource shape and codegen. That is why kernel optimization must be
planned together with the fusion.

## 2. The three-condition test, per edge

A fine-grained producer->consumer edge pays only if all three hold:

1. **Partial handover exists.** Look at the consumer's first operation on the producer's output.
   If it reduces over an axis the producer partitions (full-row norm/softmax, full-K dot product,
   accumulation before final), there is no partial readiness. Static check, minutes, no GPU.
2. **The consumer's binding engines have headroom while the producer runs.** Model each stage as a
   demand vector over MFMA/VALU issue, VMEM/LDS issue, HBM, interconnect. Two throughput-bound GEMMs
   on the same engine with a static `f/(1-f)` CU split run in `max(A/f, B/(1-f)) >= A+B` — serial.
   That rejects a *permanent role split*, not a phase-staggered CTA pipeline (tail filling, overlap
   of downstream VMEM/P2P/epilogue). A shader-issued P2P store is not free DMA; it uses waves, issue
   slots, VMEM and registers.
3. **The producer runs more than one wave.** `waves = ceil(blocks / (CUs x blocks_per_CU))`. One
   wave => early-start window ~0. Upper bound of the window: `(waves-1)/waves x producer_us`.
   Compare it with per-edge sync cost x edge count before designing anything.

Evaluate per edge, not per operator. Expect few edges to pass (an open megakernel decode layer:
1 of 6 operator pairs admitted a partial start, covering 28% of the producer).

**But an edge that fails the test may still be worth fusing for a kernel-level reason** — this is
where 1+1>2 usually comes from:
- the consumer's dependency-free operands (weights, scales, descriptors, index tables) can be
  loaded during the producer; a kernel boundary makes that prefetch inexpressible (survives even
  when the inter-kernel gap is 0);
- producer output can stay on-chip / in the same XCD L2 for its consumer (no HBM round trip, no
  wide fence);
- the partial last wave of one stage can be filled with another stage's ready work;
- the fused scheduler can choose a claim order that gives operand reuse across stages.
Write that reason down explicitly; "fuse to remove launch overhead" and "fuse so GEMM2's weight
loads issue in GEMM1's MFMA shadow" are different proposals with different implementations.

## 3. The cheaper-lever ladder (stop at the first rung that closes the gap)

0. Pick the baseline and the coverage. The baseline is the production path that serves today
   (tuned library kernels, graph capture on). The same Qwen3 MoE fusion read 1.25x against a scratch
   Triton chain and 1.05x against the tuned aiter path. The buckets to cover come from the target
   serving recipe's batch distribution (`07`), not from the UT's convenient shapes.
1. Remove work (algorithm, layout, redundant materialization, precision choice). Balance ragged
   work across CTAs and ranks (`02` B11-B12): on the Muse attention block, token-balanced attention
   alone gave 1.22x on the full layer with the original 8 launches, more than the fusion skeleton
   ever did; on the Muse MoE, rank balance was the largest lever.
2. Fuse adjacent kernels that share operands (no dependency machinery).
3. Run independent branches concurrently (multi-stream / multi-queue).
4. Graph capture (residual bubble ~300 ns per captured node; N boundaries => N x 300 ns prize).
   Measured on MI355X serving: inside a captured graph the Muse kernels ran back-to-back with ~0 gap
   (~8 µs per replay only). A launch-floor model predicted ~18.5 µs saved on the TP8
   attention-output boundary; removing 7 launches delivered ~1 µs. Do not price fusion by launches.
5. Overlap dependent consumers without going persistent (PDL on NVIDIA; ROCm has no direct
   equivalent — that gap is a legitimate argument for fusion on AMD, but make it explicitly).
6. Persistent / megakernel, only for edges that pass §2 or carry a §2 kernel-level reason.

Public calibration: adjacent fusion + concurrent branches + dependent launch typically reach 90-95%
of an ideal megakernel; restructuring the DAG beat the fine-grained mechanism on the original
structure (49.2 vs 59.4 µs); on-GPU dynamic scheduling lost on regular dense work (0.82-0.89x);
compiler tiles inside a fused kernel often lose to a tuned library kernel — that loss is charged to
the fusion.

## 4. Ordering fusion and kernel optimization

### 4.1 The order that worked
1. **Tune each stage standalone, per shape bucket.** Its tuned time is the parity bar.
   Keep the tuning in a bucket selector that the fused path can call.
2. **Fused skeleton, correctness first**, all unproven edges `whole`, each stage running its tuned
   bucket config. One allowed simplification: a coarse barrier before the final reduction/combine.
3. **Parity pass:** stage meter busy time per stage vs the standalone tuned time. Fix every stage
   that got slower (claim order, chunk, LDS lifetime, tile widening, codegen).
4. **Fusion-only levers:** cross-stage prefetch, local handoff without fences, tail filling, third
   queue, skew-gated preemption, deferred publication.
5. **Re-tune constants** for the final structure per bucket; sweep interacting constants together.
6. **Attribute** with the same tree's fusion-off control.

### 4.2 Why not "optimize standalone kernels to the end, then fuse"
- Some standalone optimizations are discarded by the fused scheduler (standalone persistence,
  spatial remap, grid-mapping reuse). Their benefit must be re-created by the fused claim order.
- The fused kernel has one resource shape; a stage tuned to use all registers standalone may force
  the whole fused kernel to 1 WG/CU.
- Constants tuned for the standalone structure are wrong for the fused one (a lever with untuned
  constants looks bad and gets rejected).

### 4.3 Why not "fuse everything first, optimize later"
- The fusion gain alone is small (0.2 ms of 4.7 ms at 8192). One expensive construct erases it,
  and a slower intermediate gets rejected by a speed gate before the levers that would pay are
  built. Separate **enabling steps** (judged on function: compiles, path marker, correct, 1000
  replays) from the **terminal step** (judged on speed, against the pre-chain baseline).
- Building overlap machinery before the stages reach parity optimizes the wrong quantity.
- Across six operators the first fused skeleton read 0.32x to 1.05x (three below 1.0x: grid
  barriers, a compiler core in place of a tuned library core, system-scope handshakes). Budget the
  rungs: one launch -> parity -> readiness edges -> levers took 12 GEAK directions and needed ~15.

### 4.4 Separate structural levers from tuned constants
- Structural lever = a switch: stage order, fused boundary, queue, handoff protocol.
- Constant = a number: tile, waves, chunk, group size, prefetch depth, reducer count.
- Expose only structural on/off as env switches; derive constants per bucket inside the selector
  and self-degrade them by budget (LDS, residency) so every batch size stays legal.

### 4.5 Keep tuned production stages outside the kernel unless they fit
A stage that production runs at the hardware floor with a tuned library kernel (expert GEMMs at
~91% of HBM on the Muse MoE) should stay a separate launch; rewriting it in another DSL to fuse it
lost 5-10% (FlyDSL) or ran at 13-19% of HBM (Triton skinny fp8). Fuse the front and tail around it,
where the fixed costs are, and say in the plan which share of the block that leaves to win.

## 5. Regimes: one structure per regime, then integrate

From the shape alone compute per guard: rows per expert/group, bytes read once (weights), FLOPs,
cross-rank bytes. The largest lower bound is the binding resource; a baseline far above every bound
means fixed costs dominate.

| regime | binding | lever family |
|---|---|---|
| fixed-cost / latency (tiny batch) | launches, handshakes, flag round trips | one launch, skip empty work, fewest round trips, no grid barrier |
| weight streaming (few rows per expert) | HBM read of all weights | read every byte once, contiguous per-WG chunks, nontemporal loads, overlap only through bytes that must be read anyway, single persistent kernel |
| compute + communication (large batch) | MFMA and interconnect | two-launch pipeline, stripe-level readiness, claim order for reuse, combine as third queue, skew preemption |

MegaMoE EP8 gfx950 evidence: single-kernel recipe 1.17-1.19x at 64 tok/rank but fails at 128/256;
two-launch pipeline loses at 64/128 (0.90/0.877x) and wins from 256 up (1.035x .. 1.22x at 8192).
A run that shared one structure across regimes held its small bucket within ~1% of baseline for 13
rounds. Do one run per regime (others are regression guards with an honest fallback), then put each
regime's path behind a dispatch rule that depends only on call-time quantities, and verify the
switch test (sizes interleaved, graph and eager, ragged ranks).

## 6. Concurrency creates an allocation problem
Once stages overlap, someone owns the CU allocation. State in every proposal: current critical path
X (N µs), expected N' µs, **the next critical path Y** and its bound. Two levers filling the same
pipe are not additive; plan the second as a follow-up. Static scheduling wins on regular work,
dynamic on data-dependent work (skewed routes); a static schedule built from averages collapses on
skew.
