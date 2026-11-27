# Fusion heuristics

Priors by workload regime. They order what to inspect first; they never decide a level on their own.

| Regime | Inspect first | Conditions and focus |
|---|---|---|
| Low concurrency | L4 MonoKernel | Only when there are many kernels, they are short, launch/sync overhead is significant, the dependency chain is strong and resource pressure is manageable. |
| High concurrency | L3 Mega / MegaMoE | Token count, per-expert batch size, GEMM utilization, dispatch/combine overhead, communication/computation overlap. |
| Medium concurrency | No preset level | Compare L2, L3 and partial L4, and decide by benchmark. This regime is the least explored. |
| Many short kernels | L1 / L2 | Remove launch overhead and HBM round trips. |
| GEMM or attention dominates | Protect the dominant operator | Keep its optimal tile, occupancy and MFMA efficiency; do not trade the dominant operator's performance for fusion. |
| Communication-heavy MoE | L3 | Overlap between dispatch, expert GEMM, combine and all-to-all. |
| Launch-bound region inside a captured graph | Single kernel | Each removed node is ~2 µs; fewer launches alone is not a reason to fuse (co-optimization `01` §3). Fuse only for an HBM round trip, on-chip handover, overlap or prefetch. |
| A stage already near its hardware floor | Keep it outside | Fuse the fixed costs around it, not the stage (co-optimization `01` §4.5). |
| Tail-latency target (P90 ITL, interactivity) under mixed scheduling (chunked prefill, speculative steps) | The longest step kind | Its effect on the tail is usually far above its device share; compute the sensitivity (SKILL Step 1, Metric response), never use the share. |
| A collective between two compute stages (TP / EP / DP / context-parallel) | Overlap | The gain is mostly the hidden communication (`overlap_hideable`), not fewer launches; gate by shape: pipeline the large shapes, keep small ones serial. |
| A region whose heavy member already has its own optimization candidate | Stacking | Build the fusion on that member's best implementation; treating them as competitors discounts both. |
| A change on a quantization or numeric path | The accuracy test's statistical power | A small-sample accuracy test can be noisier than the true difference; a verdict needs a sample size that separates them. |

## Gain estimate (co-optimization method)

- Upper bound per occurrence (fusion-only): absorbed light kernels + intermediate bytes ÷ streaming bandwidth +
  `overlap_hideable` = min(critical-path wait on the slowest rank, independent compute) × η.
- Floor first: `t_floor = min(chain, members optimized alone)`; fusion only gains below it.
- Per edge D0–D4: D3 (global reduction) and D4 (broadcast/reuse) count only with a named rewrite; else no fusion.
- Weight by covered share: the served batch / width distribution decides, not the kernel speedup
  (co-optimization `07` §3).
- Split shared (heavy member alone, both routes) from fusion-only gains; route on the fusion-only gain above the best
  split chain, per M, with the 5% / noise rule.
- Convert to the target metric with the step-kind sensitivities `s_k` (serving layer); report lower / expected / upper.
