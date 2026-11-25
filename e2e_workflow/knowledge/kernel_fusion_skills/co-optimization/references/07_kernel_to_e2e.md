# 07 — From a kernel gain to a serving gain

Source: Muse Spark attention-block and MoE mega kernels in sglang serving, MI355X TP4/EP4, DFLASH
(8 draft tokens), HiCache, InferenceX AgentX recipe, images 1027 and 1106 (2026-11). The kernel
numbers are in `05` §E-§F. This file is about why the same kernel gave +7.9% e2e on one point and 0
on another.

## 1. The chain a kernel gain has to survive

```
kernel speedup on a block        (UT: attn 1.07-1.34x, MoE 1.07-1.14x)
  x share of the forward the block is      -> per-step saving (verify step 6-16% faster at batch 3-8)
  x share of steps the kernel covers       -> batch-size distribution of the recipe
  x share of GPU time in that forward type -> verify vs draft vs prefill
  -> GPU time saved (measured 0.2% .. 5.5%)
  -> throughput only if the server is the bottleneck, else interactivity / latency
```

Each factor can be zero on its own. Measure all of them before promising an e2e number.

## 2. Predict, then confirm with a probe

- Record per-forward GPU time on rank 0 in serving (forward type, batch size, tokens, seq lens,
  gpu_ms). One line per forward is enough.
- Sweep the UT at the batch sizes and KV lengths the probe saw; predict the verify saving as
  sum(count(bs) x saving(bs)).
- Compare with the probe of the candidate server. On Muse the prediction held:

| conc | verify time at bs <= 8 | verify saving: UT prediction | verify saving: e2e probe | all GPU time saved |
|---|---|---|---|---|
| c10 | 100% | 6.4% | 5.5% | 4.8% |
| c16 | 100% | 9.4% | 7.5% | 5.5% |
| c24 | 76% | 8.3% | 5.1% | 3.5% |
| c28 | 55% | 5.9% | 5.0% | 3.3% |
| c36 | 5% | 0.4% | 0.3% | 0.2% |
| c40 | 33% | 3.1% | 2.3% | 1.5% |

Per batch it matched too (c16 batch 8: predicted 5.13 ms saved per step, measured 5.05 ms). Judge
the kernel by the probe's per-step time; judge the deployment by the official metric.

## 3. Coverage decides more than speedup
- The kernels covered verify batch <= 8 (M <= 64). Gain followed the covered share, not the kernel
  speedup: c16 +7.9% throughput / +15.2% P90 interactivity; c36 (5% covered) -1.3% / -0.1%.
- A production path at <= 16 rows (unpool fused into the next layer, no FP8 input) made both kernels
  fall back for batch 1-2, 53% of verify time at c10. Found only from the serving probe.
- Plan the bucket range from the target recipe's batch distribution before writing the kernel.

## 4. Throughput vs interactivity
- AgentX replays conversation traces at a fixed concurrency: total offered work is set by the
  client. A faster step first shows as interactivity (tokens/s/user) at low concurrency; throughput
  rises only if requests were queued, or because conversations progress further in the fixed 3600 s
  (longer contexts, more requests).
- Runs must be full length (3600 s) to compare with published numbers; throughput grows with run
  time as contexts grow.

## 5. Noise and alignment
- base vs base on the same node: throughput +-3%, P90 interactivity +-1.5%. Gains of 2-3% are not
  resolvable with one run per arm.
- A/B: run base and candidate at the same time on two 4-GPU halves, swap halves per point; start the
  second server after the first is ready (concurrent init OOMed once).
- Align the local base with the official CI before reading any gain: same image, recipe, client seed
  and metric script. On Muse the base matched CI within +-1.5% at four points. One CI point (c36)
  hit the HiCache host-cache size limit in its last 15 minutes (prefix hit rate 0.86 -> 0.5, TTFT up to
  300-700 s), a cliff local runs did not reach; drop such a point rather than chase it.
- Check for foreign GPU users before long runs; another session starting servers stalled both arms
  at once (NCCL timeouts).

## 6. The recipe and the image move under you (1027 -> 1106)
The 1106 image and recipe (InferenceX PR #4881) gave no measurable e2e gain with the same kernels:

| change | effect on the kernels |
|---|---|
| c6 / c10 moved to TP8 + AMD monokernel, c16 / c24 removed | the kernels are TP4 only; the best points disappeared |
| c36 / c40 scheduling retuned: c40 median running batch 9 -> 15 | covered share 41% -> 8% (c40), 78% / 92% of busy time above 8 requests at c36 / c40 |
| page size 16 -> 128 | the attention kernel indexed 16-token pages only |
| production verify attention / MoE got faster (image ~38-47% more throughput per GPU) | c28: same load as 1027 (median batch 7, queue ~7), but the verify step saving dropped from 6-16% to 3-11%; estimated e2e 2-3%, inside the +-3% noise |

Rules:
- Re-measure the kernel against the production path of every new image, not against the old baseline.
- Re-derive coverage from the new recipe's batch distribution (CI server metrics have
  `num_running_reqs` and `num_queue_reqs` time series).
- State which factor in §1 went to zero; "no gain" without that is not an explanation. Do not infer
  queueing from step counts; read the queue metric.

## 7. Serving constraints the UT does not show
- **Memory:** IPC buffers and per-layer state cut the KV pool by 2.4-3.6%; 32-bit KV offsets
  needed a token cap until the kernel was widened to 4 GiB pools.
- **Accuracy:** the fused attention used FP8 QKV at all M (production used bf16 at M <= 32): relL2
  3.3e-2 vs 3e-3 in the UT; GSM8K 95.00% vs 94.54% (p ~0.38, not significant). Run a task eval, not
  only relL2.
- **Distributed state:** the faster kernels exposed a production DFLASH TP desync (rank 0 draft
  tokens differed, only accept length was synced) as a deadlock in the all-reduce. The fix (broadcast
  draft tokens before verify and committed tokens after acceptance) went into both arms.
- **Graphs:** inside a captured graph launches run back-to-back; there is a ~8 µs gap per graph
  replay only. Launch-count reduction alone is worth ~0 in serving.
