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

## Gain estimate (co-optimization method)

- Upper bound per occurrence: graph nodes removed × ~2 µs + intermediate bytes ÷ 7 TB/s.
- Floor first: `t_floor = min(chain, members optimized alone)`; fusion only gains below it.
- Per edge D0–D4: D3 (global reduction) and D4 (broadcast/reuse) count only with a named rewrite; else no fusion.
- Weight by covered share: the served batch / width distribution decides, not the kernel speedup
  (co-optimization `07` §3).
