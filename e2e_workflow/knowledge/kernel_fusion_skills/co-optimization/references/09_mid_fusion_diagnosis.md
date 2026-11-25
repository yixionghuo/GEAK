# 09 — In the middle of a fusion: diagnose, pick the next lever, iterate

For the phase after the fused skeleton exists and is correct but not fast enough (Steps 4-6 of
`SKILL.md` §2). It turns measurements into the next lever. Worked example in §6: the Muse Spark
attention block went 0.685x -> 1.174x in 11 rounds, and the three largest jumps came from a
measurement, not from a new idea. MegaMoE's step-by-step reasoning is in `05` §A.1 and §B.1.

## 1. The loop (one round)

```
1. timeline   per-region end stamps, median and max over CTAs, critical path         (§2, tools/)
2. floor      per region: bytes / bandwidth, FLOPs / peak, serial hops x latency      (§3 worksheet)
3. pick       the region ON the critical path with the largest (time - floor)
4. symptom    classify it with the §4 table -> lever in `02`, rows of `04` not to retry
5. bound      expected saving <= min(time - floor, slack to the next critical path)
6. build      one lever as a compile-time arm (default off until it wins), read ISA/metadata (§5)
7. measure    paired A/B against the same tree with the arm off + re-stamp the timeline
8. record     region table, lever, result, any new "do not retry" row (§7 template)
```

Do step 2 before the first lever, not after a few rounds have failed. On the Muse attention block
rounds 8-11 went into step-loop tuning (deeper pipeline, 2 waves, software pipelining, gate GEMM on
idle CTAs) of a region that was already at ~89 of ~101 µs of its load + convert floor; every
variant was capped at ~11 µs and most lost. One floor calculation would have closed that direction.

Stop rule: 2-3 rounds in a row below the resolution floor (~2-3% per pair) on the default path
means the cheap levers are spent. Change the structure, cut bytes, or stop.

## 2. Region timeline inside one fused kernel

A fused kernel has one profiler record; the stage boundaries are gone. Put them back with
`tools/region_stamp.h` (HIP) and reduce with `tools/region_timeline.py`. FlyDSL form with the
busy/wait split: `03` §2.

### 2.1 Instrumenting a HIP kernel

```cpp
#include "region_stamp.h"          // compiled in only with -DREGION_STAMP=1
enum { S_ENTRY, S_A_END, S_B_END, S_WAIT_END, S_C_END, NSLOTS };   // a_end, b_end, ... per region

__global__ void fused(..., uint64_t* stamps) {
  RS_MARK_NOSYNC(stamps, NSLOTS, S_ENTRY);
  region_a();                       // e.g. QKV GEMM
  RS_MARK(stamps, NSLOTS, S_A_END); // __syncthreads, then thread 0 writes wall_clock64()
  region_b();                       // e.g. epilogue + publish (stamp after the release/waitcnt)
  RS_MARK(stamps, NSLOTS, S_B_END);
  wait_ready();                     // spin on another region's counter
  RS_MARK(stamps, NSLOTS, S_WAIT_END);
  region_c();
  RS_MARK(stamps, NSLOTS, S_C_END);
}
```
Host: allocate `ncta * NSLOTS` u64, launch, copy back, write `stamps_rank<r>_rep<i>.bin` (raw u64)
and one `meta.json` `{"nslots", "ncta", "tick_hz", "names"}`; `tick_hz` =
`hipDeviceAttributeWallClockRate` (kHz) x 1000 (100 MHz measured on MI355X). A persistent kernel
that runs a region once per work item stamps the end of the *last* item of that region, or uses one
slot per (region, item index) when items matter. Full example with a host side:
`tools/examples/stamp_demo.hip`.

Rules:
- `RS_MARK` puts a `__syncthreads()` before the stamp ("the whole CTA left the region");
  `RS_MARK_NOSYNC` stamps when wave 0 arrives, for wave-specialized code. Both are compiled out
  without `-DREGION_STAMP=1`; latency numbers always come from the build without stamps.
- Use `wall_clock64()` (`s_memrealtime`, constant rate), never `clock64()` (shader clock, DPM).
- A region that ends by publishing data is stamped after the release/`s_waitcnt` that makes it
  visible, otherwise the consumer's wait looks longer than it is.
- Replay **back to back** like the A/B does. A host sync after every replay puts per-rank launch
  skew into the first cross-rank wait: the Muse RS/AG read 29-36 µs that way, 7-8 µs back to back.
- Clocks of different GPUs are not comparable: compare per-rank spans and region durations, never
  stamps across ranks.
- Resolution: after a grid-wide join, CTA exit stamps spread 1.25-1.73 µs on MI355X (5 reps,
  `stamp_demo --skew`; includes release jitter, so it is an upper bound on cross-XCD skew).
  Differences under ~2 µs between CTAs are not meaningful. The first-to-last stamp under-read
  device time by ~4% and harness time by ~9% on Muse; decide with the paired A/B, explain with
  stamps.

### 2.2 Reading the table and the critical path

```
python3 tools/region_timeline.py DIR --wait wait:publish
```
`--wait R:P` declares a grid-wide edge: region R of every CTA waits for region P of all CTAs
(counter, last arriver, barrier). The critical path starts at the latest region end and walks
back; at a declared wait it jumps to the CTA whose P ended last. Output on the demo (MI355X, 256
CTAs, CTA k streams 1 + k%4 MB):

```
region          end med  end max   imbal  dur med  dur max slow CTA
entry              0.58     1.15    0.57                          5
load             278.66   464.08  186.86   278.22   463.52      127
publish          281.18   466.76  187.10     2.48     4.40      127
wait             469.72   470.59    0.92   189.64   369.48        3
reduce           472.98   473.99    1.01     3.24     4.64      116
critical path (rep with median span 473.99 us):
  CTA   67  entry        entry offset      0.56 us
  CTA   67  load         work            463.52 us
  CTA   67  publish      work              2.68 us
  CTA  196  wait         wait<-2           3.83 us  gated by CTA 67
  CTA  196  reduce       work              3.40 us
```
How to read it:
- `imbal` (end max - end median) of a producer region is tail imbalance; here 187 µs, and it shows
  up again as the median *wait* duration (190 µs) of every consumer. That is the "long wait for the
  producer" symptom, and the cause is upstream imbalance, not the wait mechanism.
- The `wait<-P` segment on the critical path is publication + polling latency (here 3.8 µs:
  max producer end -> consumer released). Large values point at sync scope / fences (`02` §C);
  small values with a large wait duration point at the producer (balance, `02` B11/B6).
- The region with the largest `work` segment on the path is the candidate for step 3 of the loop;
  compare it with its floor (§3) before choosing a lever.
- Ablation builds answer "what bounds this region" directly:
  - loads only (math removed): the load floor. Muse QKV: landed -> region end 1.8 µs without MFMA,
    4.2 µs with it, so 2.4 µs was MFMA tail;
  - no loads (math only): if still slow, the region is VALU / latency bound, not memory bound
    (Muse HIP attention: 170 µs with no loads vs 187 µs total);
  - single lever off, every lever of the build: attribution (all levers contributed in Muse r12).
- Liveness stamps: a bounded spin must record every spinner (code, observed, target), or each CTA's
  last region. Muse recorded only the first timeout; a downstream all-gather wait hid the real
  stall upstream in the o_proj split-K CTAs.

## 3. Floor of each region (worksheet)

### 3.1 Calibration numbers

Measured on this machine (MI355X, gfx950, ROCm 7.2, 2026-11-16) unless marked; re-measure on yours
with `tools/examples/hbm_read_bw.hip` and a large GEMM.

| quantity | MI355X (gfx950) | MI300X (gfx942) |
|---|---|---|
| CUs / XCDs / LDS per CU | 256 / 8 / 160 KB | 304 / 8 / 64 KB |
| HBM, vendor peak | 8.0 TB/s | 5.3 TB/s |
| HBM streaming read, measured (>= 512 MB, 16 B nontemporal, 4-8 CTA/CU) | **7.0-7.1 TB/s** | measure |
| HBM achieved inside fused kernels | 6.9 (MegaMoE weights), ~6.2 (Muse practical), 5.2 (37.7 MB stream, issue bound) | — |
| working set < 256 MB MALL, graph-replayed | partly served from MALL; region becomes issue/latency bound | same |
| BF16 GEMM achieved, torch 8192^3 | 1.27 PFLOP/s | measure |
| FP8 GEMM achieved, `torch._scaled_mm` 8192^3 | 2.66 PFLOP/s | measure |
| vendor dense peak BF16 / FP8 / MXFP4 | 2.5 / 5.0 / 10 PFLOP/s | 1.3 / 2.6 / — PFLOP/s |
| one peer copy, torch, 2 GB | 60 GB/s | measure |
| cross-rank flag round trip | ~1-3 µs (MegaMoE) | measure |
| `wall_clock64` rate | 100 MHz | read `hipDeviceAttributeWallClockRate` |

Use the measured streaming number (7.0 TB/s) as the HBM floor and treat >= ~88% of it
(~6.2 TB/s) as "at the floor". Use the achieved GEMM rate at the instruction shape the kernel
actually emits, not the vendor peak; skinny shapes reach much less (a Triton skinny fp8 core
streamed at 13-19% of HBM, `{qwen}`). Cross-rank bandwidth with kernel-issued stores to all peers
at once is not the single-peer copy number; measure it with the kernel's own pattern.

### 3.2 Steps

For every region on the timeline:

1. **Bytes that must move**, split by where they come from:
   - HBM reads needed once: weights + their scales, activations, KV (+ scales, page tables),
     metadata tables. Count each tensor once; duplicates are a finding (§3.3).
   - Writes: outputs, partials, scratch; plus the re-read of partials by their consumer.
   - On-die only (working set < MALL and replayed in a graph, or produced and consumed on the same
     XCD within the region): mark them; they do not count against HBM.
   - Cross-rank bytes: payload per peer, per direction.
2. **Math**: FLOPs at the emitted MFMA dtype/shape (2·M·N·K per GEMM), plus VALU work that is not
   hidden (softmax exp, convert, norm) — count VALU ops per element if the no-load ablation is
   still slow.
3. **Serial hops**: dependent steps that cannot overlap (load round trip -> compute -> publish ->
   remote poll), each x its latency (HBM load ~1-2 µs under load, flag round trip, xGMI round trip).
4. **Floor** = max(HBM bytes / 7.0 TB/s, FLOPs / achieved rate, link bytes / link rate)
   + serial hops x latency (only for hops that nothing overlaps).
5. **Efficiency** = floor / measured region time (critical-path segment from §2), and
   **gap** = measured - floor.
6. Decide:
   - efficiency >= ~0.85-0.9: only byte cuts pay (symptom S6 in §4). Close reordering, prefetch
     depth, extra waves and software pipelining for this region and record why.
   - large gap but region not on the critical path: no gain until its slack is gone.
   - pick the critical-path region with the largest gap; expected saving <= min(gap, slack to the
     next critical path).

### 3.3 Two checks that found the largest losses
- **Bytes fetched vs bytes needed.** Muse HIP attention fetched each 256 B K/V row once per 64 B
  head slice (4 waves walking the same segment): 2x over-fetch, 153 -> 122 µs once fixed. Compare
  `FETCH_SIZE` (rocprofv3 `--pmc FETCH_SIZE`, KB) or achieved bytes/time with step 1 before tuning
  anything else.
- **Near the floor -> only byte cuts pay.** At 89 of 101 µs, every reordering of the attention step
  loop (software pipelining, prefetch depth, 2 waves) was capped at ~11 µs and all of them lost.

### 3.4 Worked floor: MegaMoE weight streaming, 64 tok/rank (gfx950)

```
weights per expert  W1 22 MB + 0.69 MB scales, W2 11 MB + 0.34 MB scales = 34.0 MB
active experts      48 per rank (all, at 64 tok/rank: 48*(1-(1-1/48)^384) ~ 48)
HBM bytes           48 x 34.0 MB = 1.63 GB (1.68 GB with MiB rounding)
math                ~792.7 MFLOP x 64 = 51 GFLOP  -> ~10 µs at MXFP4 rates: not binding
floor               1.63-1.68 GB / 7.0 TB/s = 233-240 µs   (weights read once)
measured            ~291 µs graph time
gap                 ~51-58 µs:  front end -> GEMM1 start 20-21 µs (P2P write floor 11-12 µs)
                                per-XCD GEMM1 spread 36-40 µs, slow XCD changes per run
                                combine tail ~4 µs after the slowest rank
decision            G1+G2 at ~1.12x of the HBM ideal: no extra reads, no deeper prefetch
                    (L1, L4); the remaining levers are front-end latency and XCD imbalance.
```

## 4. Symptom -> lever -> do not retry

Columns: what you see (from §2 / §3), what to check first, levers in `02`, rows of `04` not to
retry (`L` = ideas that lost, `T` = structural traps, `H` = hang checklist). The first six rows are
the common mid-fusion symptoms.

| # | symptom | how you see it | first check | levers (`02`) | do not retry (`04`) |
|---|---|---|---|---|---|
| S1 | a region is slower than its standalone tuned kernel | region `dur` (§2) > standalone kernel time at the same shape; stage meter busy > baseline | bytes fetched vs needed (§3.3); tuned tile/unit config carried over?; claim order vs original grid mapping; spills (§5) | E1-E2 tuned configs, B1-B3 claim order and heads, G1 wave per KV head, G10 loads first, F1-F3 codegen | L39 rewrite tuned stage in another DSL, L7 smaller tiles, L21 strided shard, L20 nested loop |
| S2 | long wait for producer readiness | `wait` region long; critical path jumps to a producer CTA (`wait<-P`) | split it: producer tail (`imbal` of P large) vs publication latency (`wait<-P` segment large) vs consumer idle with nothing issued | tail: B11 balance, B6 tail shaping, LPT order; publication: C2-C3, C6, C8, C9; idle consumer: A1, A6, G9 issue dependency-free loads before the wait, B7 fill while waiting, B9 phase-staggered queues | L27 fine readiness on balanced routes, L28 speculative claim, L15 ungated preemption, L31/L45 grid barriers, L5 claim-ahead |
| S3 | long cross-rank wait | wait on a peer flag dominates; per-rank spans differ | host-sync artifact (§2.1, replay back to back); no-payload control (`03` §1); link bytes vs round trips | B10 publish once per destination, B12 rank balance, B14 rotate peers / push, C10 buffer stores, C11 data as flag, A11 reductions inside the comm | L41 helper CTAs gated by remote arrival, L42 pull-mode, L36 one-launch tail, L8 pre-open gate |
| S4 | spills, scratch, VGPR+AGPR at the limit | `.vgpr_spill_count`, `.private_segment_fixed_size` > 0, `scratch_` in the hot loop (§5) | which live range: hoisted constants, descriptors in the outer frame, nested loop, default-off arms | F1 anchor constants, F5 no nested loop / descriptors in unit frame, F11 compile out arms, F2 uniform values in SGPR, E3 LDS by lifetime | L3 deeper buffering, L46 continuous ring, L25 cutting LDS when VGPR binds, L33 2 waves/SIMD, T9 static screen as proof |
| S5 | idle CTAs at the tail or during a region | `imbal` large on the last region; high CTA ids with no item; stamps end early | is the region HBM bound (§3)? idle CTAs cannot add bandwidth | B6 tail shaping, B11 balance, B2 chunk size, G8 merge units on idle CTAs, A9 reduction as third queue | L34 weight stream on idle CTAs (+16%), L2 stealing across XCDs, L24 larger chunks |
| S6 | region at the HBM ceiling | §3 efficiency >= ~0.85-0.9 (>= ~6.2 TB/s achieved on MI355X) | is every byte needed: duplicates, format, precision of partials, over-fetch | B10 dedup, G7 bf16 partials, data format, G1 over-fetch fix, keep reused weights in L2 (B1, not L22) | L51 software pipelining, L52 deeper prefetch, L4 lookahead 3/4, L33 2 waves, L1 / L35 / L55 extra early reads, L48 cross-unit prefetch |
| S7 | time between regions; whole-grid sync | gap between one region's max end and the next region's start on every CTA | grid barriers left on the default path? | C9 readiness edges, G11 last arriver, A6 hoist setup | L31 barrier-phased skeleton, L19 / L32 "one launch" as goal |
| S8 | long chain of small dependent steps at 1 CTA/CU | many short regions on the critical path, each ~1-2 µs | count hops x latency (§3.2 step 3) | G10 issue all loads before math, G2 fewer longer pieces, G12 single-round merge, A11 split-K inside the comm | more CTAs per item; finer pieces |
| S9 | prefetch or weight stream gives no net gain, or only moves time between regions | A/B neutral; stamps show one region faster and the next slower | does it share the in-order `vmcnt` with latency-critical loads (§5)? | A5 issue order, G9 split arrive/wait around the stream, separate waves or direct-to-LDS | register prefetch from the same waves, L55 warm-up loads, L53 prologue prefetch before the epilogue |
| S10 | `vmcnt(0)` / `readfirstlane` inside the hot loop | ISA of the loop (§5) | conditional reload into loop-carried slots; uniform id loaded by VMEM | F10 unconditional loads, G6 clamped reload, G5 page ids in LDS as vector values | L57 `readfirstlane` on a VMEM-loaded index |
| S11 | step body VALU bound at 1 wave/SIMD | no-load ablation still slow (§2.2) | VALU per step; register count | G4 scale folding, denominator by MFMA, F12 wider MFMA, G3 static softmax (check headroom) | L54 branchless step body, L33 2 waves without a register diet |
| S12 | hang or timeout after a structural change | bounded spin record, last region per CTA | which spinner, observed vs target, for every spinner | `04` H1-H12; ISA diff of the stalled region; scratch aliasing between regions | retrying the change without the root cause; L38 program_id ordering |
| S13 | lever wins in isolation, loses in the build | A/B of the build vs single-lever-off | registers or vmcnt order changed elsewhere (§5 metadata diff) | re-read metadata; single-lever-off attribution | T2 judging coupled levers one at a time, L43 runtime-off arm compiled in |

## 5. Reading ISA and resource metadata

Read after every structural change. All commands below were run on ROCm 7.2 / gfx950; replace
`gfx950` with `gfx942` on MI300X.

### 5.1 HIP

```bash
# resource usage at compile time: VGPR, AGPR, SGPR, scratch, occupancy, spills, LDS per kernel
hipcc -O3 --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -c k.hip -o k.o
#   remark: VGPRs: 24 / AGPRs: 0 / ScratchSize [bytes/lane]: 0 / Occupancy [waves/SIMD]: 8 /
#           SGPRs Spill: 0 / VGPRs Spill: 0 / LDS Size [bytes/block]: 1024

# keep the assembly and the device ELF
hipcc -O3 --offload-arch=gfx950 --genco -save-temps k.hip -o k.co
#   k-hip-amdgcn-amd-amdhsa-gfx950.s    assembly
#   k-hip-amdgcn-amd-amdhsa-gfx950.out  device code object (ELF), readable directly
# --genco output is an offload bundle; unbundle it before readobj/objdump:
/opt/rocm/llvm/bin/clang-offload-bundler --unbundle --type=o --input=k.co \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=k_gfx950.co

# metadata of a code object (.vgpr_count .agpr_count .sgpr_count .vgpr_spill_count
# .sgpr_spill_count .group_segment_fixed_size .private_segment_fixed_size per kernel)
/opt/rocm/llvm/bin/llvm-readobj --notes k_gfx950.co | grep -E "\.name:|_count|segment_fixed_size"

# code objects embedded in an executable or a .so (torch extension, aiter module):
/opt/rocm/llvm/bin/llvm-objdump --offloading lib.so     # writes lib.so.0.hipv4-amdgcn-...-gfx950
#   (roc-obj-ls / roc-obj still work but are deprecated)
# prebuilt .co / .hsaco files (aiter hsa/ kernels) are code objects already: readobj them directly

# disassembly, optionally with source lines (build with -gline-tables-only)
/opt/rocm/llvm/bin/llvm-objdump -d -l --mcpu=gfx950 k_gfx950.co > k.dis
```

### 5.2 Triton (AMD backend, Triton 3.x)

```python
c = kernel[grid](...)                 # returns the CompiledKernel
c.asm["amdgcn"]; c.asm["hsaco"]       # also "ttir", "ttgir", "llir"
c.n_regs, c.n_spills                  # VGPRs and spills of this specialization
c.metadata.shared, c.metadata.num_warps, c.metadata.waves_per_eu
```
- Cache directory (`TRITON_CACHE_DIR`, default `~/.triton/cache/<hash>/`): `<kernel>.amdgcn` (with
  the same `.vgpr_count` / `.agpr_count` / `.vgpr_spill_count` / `.group_segment_fixed_size`
  metadata block as HIP), `<kernel>.hsaco` (readobj/objdump it like §5.1), `<kernel>.json`
  (launch metadata), `.ttgir`, `.llir`.
- `AMDGCN_ENABLE_DUMP=1` prints the assembly to **stdout**, and only when the kernel is actually
  compiled: empty the cache or set `TRITON_ALWAYS_COMPILE=1`, otherwise it prints nothing.
- Each constexpr / specialization is a separate binary; read the one the timed run launched.

FlyDSL: GEAK `knowledge/gfx950_lowering.md` (COMPILE_ONLY=1 ISA dump; profiler VGPR column is in
units of 2 on gfx950 — read `.vgpr_count` from the code object instead).

### 5.3 Occupancy from the metadata

```
regs per wave   = roundup(vgpr, 4) + agpr  (unified 512-entry file per SIMD lane on gfx90a/942/950;
                  allocation granule 8)
waves per SIMD  = min(8, floor(512 / roundup(regs per wave, 8)))
WGs per CU (reg)= floor(4 * waves per SIMD / waves per WG)          (4 SIMDs per CU)
WGs per CU (LDS)= floor(LDS per CU / group_segment_fixed_size)       (160 KB gfx950, 64 KB gfx942)
WGs per CU      = min of the two (and the hardware WG limit)
```
Example: 256 VGPR + 0 AGPR, 512 threads (8 waves) -> 2 waves/SIMD -> 1 WG/CU, whatever the LDS.
Cross-check with the `Occupancy [waves/SIMD]` remark. A persistent grid must be <= WGs per CU x CUs
(residency is correctness, `02` E5).

### 5.4 Finding and reading the hot loop
- The loop is the block between a label and a backward `s_cbranch_*` (objdump prints the backward
  offset as a large unsigned number, e.g. `s_cbranch_execnz 65520`, and the target label).
- Grep inside it:
  - `s_waitcnt vmcnt(0)`: a full drain. `vmcnt(N)` waits until at most N loads are outstanding,
    in issue order, so it also drains every load issued before the one you meant (prefetches
    included). Find what forced it: a phi copy of loop-carried load registers (conditional reload),
    `v_readfirstlane` of a loaded value, a conditional load.
  - `v_readfirstlane_b32` after a VMEM load: forces a wait on that load and everything before it.
  - `scratch_load` / `scratch_store` (or `buffer_*` on the private segment): spills.
  - `v_accvgpr_read` / `v_accvgpr_write` storms: AGPR copies around MFMA.
  - `global_store` / `buffer_store` to peer memory followed by `s_waitcnt vmcnt(0)`: one link round
    trip per store.
  - the MFMA variant actually emitted (`v_mfma_f32_16x16x128_f8f6f4` vs four `16x16x32`), and the
    number of `s_barrier` per K block.

### 5.5 Checking that a disabled branch did not change the kernel
A runtime-off arm still occupies registers and can change scheduling (+1.4 µs on `{attn-out}`, L43).
Compare three builds of the timed instance: pristine, arm compiled out, arm compiled in but off.
`sha256sum` of the disassembly (or of the `.out`) of "compiled out" must equal pristine; diff the
metadata (`vgpr_count`, `agpr_count`, spills, `group_segment_fixed_size`) and the hot-loop
instruction counts of "compiled in, off" against it. Any difference means the arm must be a
compile-time switch.

## 6. Worked example: Muse Spark attention block (TP4, gfx950), 0.685x -> 1.174x

| R | what the measurement showed | diagnosis | lever | geomean after |
|---|---|---|---|---|
| 1 | F56 attention bound by the longest request (equal split per request) | imbalance | token-balanced pieces (8 launches still) | F56 1.22x alone |
| 2 | one launch, grid barriers: attention 187 vs 116 µs; no-load build still 170 µs; S56 front + tail 80 µs > whole 70 µs baseline | region inefficiency + barriers; fusion measured nothing yet | none (skeleton) | 0.685 |
| 3 | RS/AG 48/37 µs only with host sync (7-11 µs back to back); 4 waves walked one segment | measurement artifact; 2x over-fetch | wave = KV head, full rows; static softmax 177 -> 161 µs; QKV in 256 equal units 29 -> 12 µs | 0.859 |
| 3 | region table: 4 barriers ~18 µs at S56, ~6 µs each | barriers now the binding cost | — | |
| 4 | barriers -> counters ~7 µs; Wo prefetch in any position: no net change | Wo shares the in-order vmcnt with the merge | readiness counters; QSMIN=4 (partials as large as K/V at 2 steps) -5 µs | 0.937 |
| 5 | ISA: `vmcnt(0)` every NPF steps in the K/V loop | conditional reload made the slots a phi | unconditional clamped reload 125.7 -> 114.5 µs; page ids in LDS; scale folded into Q; last barrier removed; arrive/wait split around Wo 164.6 -> 157.1 µs | 1.064 |
| 6 | merge 11 µs vs 3.7 µs baseline; high CTA ids idle | merge on the critical path, idle CTAs | merge units on idle CTAs; bf16 partials (half the bytes) | 1.081 |
| 7 | o_proj tail 10+ µs after the last slab | serial o_proj + RS chain | split-K x2 exchanged in the RS; single-round sliding merge | 1.118 |
| 8 | attention per-CU throughput bound | load pipeline depth | deeper K/V pipeline, warm page ids | 1.156 |
| 9 | register report 255 VGPR + 160 AGPR; loads + convert 89 of 101 µs | at the byte floor; 2 waves infeasible | (SWP lost in all variants) -> stop step-loop levers | 1.149 |
| 10 | full timeline: F56 attention 96 of 150 µs; S56 ~144 CTAs idle 18-30 µs | F floor-bound; S has idle CTAs | (gate GEMM deferral hung) | 1.155 |
| 11 | gate weights on idle CTAs: S56 +16% | HBM contention: S region was not compute-idle | dead end recorded | 1.150 |
| 12 | weight issue at 2.1-2.9 µs (serial segment table); loads-only ablation: MFMA tail 2.4 µs | issue latency + MFMA tail on the head of the critical path | loads at entry, lane-parallel table, 16x16x128 MFMA, epilogue prefetch: ~4 µs, 1:1 at the tail | **1.174** |

What to copy from it:
- Rounds 3 and 5 were won by reading the data (artifact, over-fetch, ISA), not by trying levers.
- Round 9 saved the rest of the budget from step-loop tuning by computing the floor.
- Rounds 4, 10 and 11 show the two costly misreads: a prefetch that "should overlap" but shares
  the vmcnt queue, and idle CTAs that are idle because the region is memory bound.

## 7. Round log template

```
round N   build <sha>   default path = <arms on>
region | median end | max end | dur on path | bytes / FLOPs / hops | floor (kind) | efficiency | gap | critical path? | note
-------+------------+---------+-------------+----------------------+--------------+------------+-----+----------------+-----
a qkv  |            |         |              |     |                |
...
binding now:   <region>, <symptom S# of §4>
lever:         <02 id>, arm <flag>, expected saving <= <µs> (gap) and <= <µs> (slack)
not retried:   <04 L/T/H ids checked>
result:        paired A/B <ratio> (n, wins, null), re-stamp <region deltas>, ISA/metadata <vgpr, spills, vmcnt>
new dead end:  <row for 04, with the measurement>
```
