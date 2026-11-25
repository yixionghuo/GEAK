# 04 — Measured losses and pitfalls (do not retry without a new reason)

Rows without a source mark were measured on MegaMoE EP8, gfx950, FlyDSL. Marked rows come from
`{attn}` Muse attention block (gfx950, HIP), `{moe}` Muse MoE (gfx950, Triton/HIP), `{attn-out}`
Muse attention-output (gfx942, HIP), `{qwen}` Qwen3 MoE (gfx942, Triton/HIP); see `02` header.
Mechanisms marked `[AMD]`, `[gfx950]`, `[multi-rank]` do not carry over unchanged to other targets;
on another target a row is a hypothesis to re-test, not a ban. A row seen on several operators
(early extra weight reads, grid barriers, launch-only fusion, rewriting tuned stages) is the
strongest evidence in this file.

Rows are numbered (`L<n>` for losses, `T<n>` for structural traps, `H<n>` for hang checks) so the
symptom table in `09` §4 can point at them.

## Ideas that lost

| id | idea | result | why |
|---|---|---|---|
| L1 | any extra early HBM read in the front end (B prefetch before plan, W1 warm-up, G1 preload, MALL prefetch) | +1..+10 µs @64 | competes with payload writes; overlap only through required bytes |
| L2 | work stealing across XCDs (producer or consumer units) [AMD] | +1.3..+8 µs @64 | re-reads A from HBM, needs L2 writeback/acquire |
| L3 | deeper whole-tile B buffering in GEMM2 | +2-3 µs, VGPR spills | register budget |
| L4 | lookahead 3/4 | +1..+4 µs | issue/register cost |
| L5 | claim-ahead at unit start | +1..+4 µs | removes the claim gap but adds no bytes in flight |
| L6 | claim next unit at K-step 20 instead of 24 | slower | too early |
| L7 | GEMM2 BN128/BK128 inside fused kernel | +5 / +3.7 µs | smaller tiles lose reuse |
| L8 | previous launch pre-opens next gate [multi-rank] | +4..+7 µs | hot epoch word |
| L9 | system release per CTA before exit ticket [gfx950] | +54 µs | L2 writeback per CTA |
| L10 | agent release per GEMM1 tile + XCD-interleaved GEMM1 [gfx950] | single kernel 75-100 µs slower | fence per tile |
| L11 | CTAs past the work bound take exit tickets | hundreds of µs | one hot atomic |
| L12 | write-through GEMM1 stores (XCD-local recipe) [gfx950] | +10 µs @64 | not needed when handoff is same-XCD |
| L13 | two lockstep 4-wave Stage2 halves | +0.53 ms @8192 | use one NW8 slab |
| L14 | per-block L2 writeback instead of write-through P2P [gfx950] [multi-rank] | +0.41 ms @8192 | fence cost |
| L15 | skew preemption active on uniform route | +0.30 ms @8192 | machinery with nothing to overlap; gate it |
| L16 | quantization work-stealing atomics | +0.30 ms @8192 | claim atomic on every block's path |
| L17 | work-stealing claim queue for a uniform in-kernel phase | ~89 ns per chunk (+0.30 ms @8192) | static grid-stride split beat it |
| L18 | quant pass folded into persistent kernel as static partition | +16.5 µs (8192 u) .. +65 µs (8192 skew) | software max-of-N join replaced a hardware join |
| L19 | "one launch" as a goal | slower on all four guards, withdrawn | phase < 1% of e2e |
| L20 | nested while around GEMM2 body | ~2.6x | loop-carried live range spills |
| L21 | 4-way strided shard of consumer head | +3.7% @8192 | breaks m-block locality |
| L22 | W2 loads bypassing L2 (`sc1`) [gfx950] | +5.5% | W2 reuse lives in L2 |
| L23 | 128 payload rows (dispatch) [multi-rank] | +40% | dispatch geometry; 256..384 flat |
| L24 | larger G2 chunks with column-grouped order | C16 4.880 vs C1 4.089 ms | tail quantization |
| L25 | cutting LDS to reach 2 WG/CU when VGPR=256 [AMD] | +2.33 ms | wrong binding resource |
| L26 | grid-stride unroll U=4 vs U=1 | 4.4535 vs 4.4476 ms | residual not latency-bound |
| L27 | cross-block early GEMM2 entry (fine readiness) at 1 WG/CU on uniform | slower than coarse floor | little window on balanced routes, added atomics |
| L28 | destructive speculative claim of unready consumer work | livelock | use non-destructive peek |
| L29 | per-row fine GEMM2 / combine interleave [multi-rank] | net loss, deadlocked twice | use the third queue |
| L30 | combine reducers beyond per-XCD residency (two-kernel path) | deadlock | residency |
| L31 | phases separated by grid barriers inside one launch {attn} | 0.685x; 4-6 µs per barrier at 1 CTA/CU | whole-grid join; use readiness edges |
| L32 | removing launches as the goal {attn-out} {moe} | 1.016x (model said ~18.5 µs, got ~1 µs); MoE launch-only fusion ~0 | in a captured graph kernels already run back-to-back |
| L33 | 2 waves/SIMD for attention {attn} | blocked (415 regs > 256); cross-step SWP lost in every variant | register wall |
| L34 | defer a weight-streaming GEMM into idle CTAs during attention {attn} | S56 +16% slower | HBM contention with the K/V stream; region already at its load floor |
| L35 | prefetch expert weights during the front {moe} {qwen} | 0.31 µs/MB cost vs 0.12 µs/MB saved; +2..10 µs | no idle HBM window; overlap only through bytes that must be read anyway |
| L36 | one-launch tail (unpool + all-gather fused) {moe} | 0.4-8.5% slower; tail push in epilogue -0.64% | tail is xGMI-latency bound, not launch bound |
| L37 | fold finalize rows into the expert GEMM {moe} | -2.2% at M56 | Triton row body ~9 µs slower than the HIP finalize |
| L38 | G1+G2 dependent grid ordered by program_id {moe} | deadlock (M32 hang) | program_id order does not guarantee residency; use ticket claim |
| L39 | rewrite tuned production expert GEMMs in another DSL to fit the fused kernel {moe} {qwen} | 5-10% slower; 13-19% of HBM | a compiler tile inside a fused kernel loses to a tuned library kernel; charge it to the fusion |
| L40 | full expert replication for EP balance {moe} | 1.18x but +5.1 GB per rank per layer | not deployable; use partial redundancy |
| L41 | helper CTAs for a phase gated by cross-rank arrival {attn-out} | -15% | the phase waits on remote RS arrival, not on local work |
| L42 | pull-mode all-gather {attn-out} | -30% | 2.6x remote bytes plus remote polling; push instead |
| L43 | default-off arm code compiled into the hot instance {attn-out} | +1.4 µs | changes register allocation; compile arms out |
| L44 | per-row-block RS push pipelining after loads overlap {attn-out} | negative | in-order vmcnt |
| L45 | whole-grid G1 -> G2 barrier with system-scope tickets {qwen} | 0.837x, ~71 µs handshakes | sync scope too wide |
| L46 | continuous cross-item weight ring / barrier-decoupled MFMA {qwen} | 3-6 µs slower / neutral with 43 spills | register budget |
| L47 | Stream-K for the skinny G1 {qwen} | 0.891x (hybrid 0.968x); mixed-mode process faulted | counters shared between instances |
| L48 | cross-unit L2 weight prefetch {qwen} | 0.945-0.993x, worse with depth | in-order vmcnt plus XCD L2 overflow |
| L49 | move the act quant into the G2 A-load {qwen} | 0.962x / ~0.79x | work moved onto the critical path |
| L50 | per-expert staged readiness at T=64 {qwen} | -2..-4.4% (kept at T<=32) | machinery with little window on balanced routes |
| L51 | software pipelining across attention steps (SWP, sched_group_barrier) {attn} | 101.3 -> 102.6..110.5 µs; AGPR 160 -> 176-192 | region at ~89% load floor |
| L52 | prefetch depth NPF 3/4, o_proj PD 3/4, merge batch 32 {attn} | no gain or -1..-2% | not the binding resource |
| L53 | attention prologue prefetch before the QK epilogue {attn} | F56 attention 190 µs | competes with the epilogue loads |
| L54 | branchless attention step body {attn} | +50 µs | +~128 VALU per step at 1 wave/SIMD |
| L55 | L2/MALL warm-up of Wo {attn} | S56 -1.3% | extra read |
| L56 | `-amdgpu-mfma-vgpr-form`, asm register pinning {attn} | 0 µs | accvgpr copies were not the cost |
| L57 | `v_readfirstlane` on a VMEM-loaded page id {attn} | +50 µs at F56 | forces vmcnt, drains all earlier K/V prefetch |

## Structural traps (T1..T15)

1. **Half replacement.** Grafting the new readiness/peek primitives on top of the old coarse-barrier
   body keeps both regimes live: old barriers still serialize, new atomics are pure overhead
   (8.14 ms vs 7.4 ms floor). Delete the old body. Tell: drain still contains coarse/barrier code
   and zero peek sites.
2. **Coupled levers measured one at a time.** Levers that only pay together (real producer/consumer
   co-residency + fine readiness + deeper pipeline) each fail an isolated A/B. Plan coupled levers
   as one build with one activation set, measured together, with intermediate steps judged on
   function.
3. **Constants exported as switches.** Large-batch constants forced onto small-batch correctness
   runs (LDS overflow, wave-count asserts). Derive per bucket.
4. **Silent fallback.** A fast path gated on `ENV==1 && predicate` measured in its slow form. Print
   the path once per process; refuse numbers without the marker.
5. **Fusion deletes its own instruments.** A wait timer in a kernel that is no longer called reads 0.
6. **Summing stage timers after overlap** — moves the wrong way (62.4 -> 102.9 µs summed while span
   fell 61.6 -> 37.0 µs).
7. **Reading a whole-grid-join arm as a candidate.** It is the control (cost of unification). It
   tends to win on imbalanced routes (idle time absorbs the cost) and lose on balanced ones.
8. **A route gate resolved once and cached under graph capture** picks the path for all later
   batches. Shippable form: one graph per path + host-side per-batch statistic.
9. **Static ISA screen used to close a scheduling direction.** ISA unchanged does not mean no effect
   (an index-scalarization + prefetch-lead change was ISA-identical and 0.4-1.6% faster).
10. **Lease-only kernels.** A body that calls the comm runtime at trace time cannot be compile-only
    screened; find out on day one and plan instrumented multi-question runs.
11. **Fused kernel fixed to a batch range the serving recipe leaves.** A kernel covering verify
    batch <= 8 gave +7.9% e2e at c16 and ~0 at c36 / c40, where most batches were larger. A new
    recipe on a newer image moved c40's median batch from 9 to 15 and the covered share from 41% to
    8%. Build coverage from the recipe's batch distribution, and re-check it at each recipe change.
12. **A production path you did not know about.** At <= 16 rows production fused the unpool into the
    next layer's boundary and had no FP8 input, so both mega kernels silently fell back for batch 1-2
    (53% of verify time at c10). Log the path taken per batch size in serving, not only in the UT.
13. **Faster kernels expose latent distributed bugs.** A DFLASH TP desync existed in the base
    (draft tokens differed on rank 0; only accept length was synchronized). The mega kernels changed
    timing and turned it into a deadlock inside the production all-reduce. Run a token-consistency
    check across ranks, and run base and candidate with the same fix.
14. **Host checks run only at CUDA graph capture.** Capacity and argument checks in the Python
    wrapper do not run on replay; the kernel must bound itself (trap with an error record), and the
    caller must keep replays within the captured limits.
15. **Profiler-induced rank skew.** `torch.profiler`-based timing starts ranks at different times;
    a cross-rank kernel then spins for the skew. Spin bounds must exceed it, and tests barrier
    before each eager call.

## Hang / fault checklist (in order, H1..H12)
1. Residency from emitted metadata (VGPR+AGPR, LDS x WG/CU) vs grid; one added value can cross it.
2. Every waited/atomically updated address inside its allocation for the largest size and grid.
3. No clear of a word another CTA may already count into in the same launch.
4. A waiting unit is never claimable before all units it waits on are claimed by resident CTAs;
   claim mapping is a bijection.
5. Monotone words waited with `>=`, never `==`.
6. Coordinator owes workers nothing after its coordination point (acyclicity).
7. Per-launch metadata readers all wait on that launch's readiness flag (not only one role).
8. Producer/consumer share one index and one payload format derived from one source.
9. Deterministic fault on every rank that persists when the new store becomes a benign constant
   store => codegen/regalloc frame fault; shrink live ranges, pin bases to SGPR, bound buffer
   resources to real extents.
10. Grid must be fully resident before any CTA waits on another (dependent-grid launches ordered by
    program_id are not); assert residency from emitted metadata, claim work by ticket.
11. Every rank must issue the same sequence of calls on an instance (IPC rings keyed by call
    counters); a rank that diverges (different batch, skipped call) hangs the peers. Check the
    serving scheduler's batches are identical across TP ranks.
12. Bounded spins with an error record (site, epoch, observed value) and a trap; a record-only build
    (no trap, short bound) for diagnosis.
