# Report contents

Deliver model notes (Step 0), a report with the sections below, and the machine-readable list of all
candidates in the format the caller defines.

## Workload characterization

- concurrency
- batch / token distribution
- prefill / decode split
- parallelism
- dominant bottleneck
- average kernel duration
- launch overhead
- memory traffic
- communication
- resource utilization

## Metric response

The Step 1 table: one row per step kind (device share, duration p50 / p90, tail exposure, queue exposure,
sensitivity `s_k`), the source of every `s_k`, and the ones that are unknown (listed under missing information).

## Boundary ledger

The per-phase kernel sequences and, for every adjacent boundary, the covering candidate or the reason
it is not fused (Requirement 5).

## Candidate fusion patterns

| Rank | Level | Fusion region | Kernels (trace names) | Launches before → after | Expected gain (target metric %, lower / expected / upper) | Shared gain (%) | Fusion-only gain (%) | Single-kernel route (expected %) | Route per M | Main risk | Conflicts / composes with | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Fusion against the single-kernel route

For each candidate, the single-kernel estimate, where its member's plausible speedup comes from, and
the route per M (Step 5). List the candidates that go to the single-kernel route and why.

## Attribution

For every recommended candidate: the shared gain (the heavy member alone, both routes) and the fusion-only gain
with its three terms (absorbed kernels, intermediates off HBM, `overlap_hideable` with its η), and the single-kernel
candidates it composes with (and what merging them needs).

## Calibration

Completed measurements of this run or of records of the same deployment and workload: estimated interval against
measured gain per candidate, and the correction applied to `s_k` or η because of them. Empty when nothing was
measured yet.

## Recommended fusion patterns (implementable scope)

For each recommended pattern state the fusion boundary, the level, the reason for the choice, the
bottleneck it addresses and its seam. The recommended patterns are mutually compatible.

## Why not other levels

For each recommended pattern, why a smaller scope was not chosen and why a larger scope was not chosen.

## Required validation

What must be confirmed before implementation: dependencies, tensor shapes, VGPR/LDS, occupancy, tile
configuration, communication, correctness and e2e performance.
