# 08 — Shipping a persistent cross-rank kernel as a library op

Source: upstreaming the Muse Spark attention-block kernel to aiter (PR #7870) and the MoE tail
all-gather, 2026-11, plus several review rounds. These are review findings that a GEAK candidate
passes without noticing, because its UT only checks the happy path.

## 1. Placement
- HIP kernels go to the kernel library (aiter); model integration goes to the serving framework
  (sglang) as a separate PR. Split independent kernels into separate PRs.
- Keep the op opt-in: nothing dispatches to it unless the caller constructs it.

## 2. Host-side checks
- Explicit exceptions (`ValueError`, `NotImplementedError`), never `assert` (stripped by `python -O`).
- Check every tensor the kernel indexes: device, shape, dtype, contiguity, layout (KV pool shape,
  block-table row stride, scale views), finite positive scalars.
- **Per-call checks must depend only on values identical on every TP rank**, so all ranks refuse
  the same call and none enters the kernel alone. Value checks on indices (pages in range, slots,
  positions) are collective and opt-in (`validate_indices=True`, all-gather of the verdict).
- Check the bound device, not the current device (arch, CU count); run the launch under
  `torch.cuda.device(self.device)`.
- Numeric headroom: a static softmax (no running max) needs
  `max_log2_score < 126 - log2(max_keys) - log2(448)` for an FP8 V; check it from q_scale / sm_scale.
- Capacity limits are named constants checked on the host (`MAX_KV_STEPS`, `MAX_WINDOW`, the page-id
  table size `PID_MAX`), with an optional host-side bound (`kv_tokens`) for eager callers.

## 3. Collective lifecycle
- Setup is collective: record each rank's failures (build, alloc, IPC open), all-gather the result,
  raise on every rank, release partial state (close peer mappings, barrier, then free).
- `close()`: synchronize, barrier, close peer handles, barrier, free, unload.
- Document what remains NCCL-like: a rank failing after the checks (OOM, launch error) leaves peers
  waiting until the spin bound.

## 4. In-kernel bounds
- Every spin is bounded (600 s in production builds) and records site, epoch and observed value
  before trapping. A record-only build (`FAIL_TRAP=0`, short bound) is the diagnosis tool.
- The bound must exceed per-rank start skew; `torch.profiler`-based timing (`run_perftest`) skews
  rank starts.
- Invariant violations (out-of-range page id, steps above table size) trap with a code.
- Inside a CUDA graph only the kernel's own bounds protect replays; say so in the docstring.

## 5. Tests (library UT standard)
- The library's own UT style: `@benchmark`, `run_perftest` with rotating copies of weights / KV /
  inputs (cold L2), a markdown summary table, exit non-zero on failure, skip on other archs.
- Check every output: main output, new residual stream, the KV written, FP8 copy and its per-row
  scale; graph replays with alternating inputs.
- Edge profiles: context lengths around page and window edges, a small window where one missing key
  moves the output (~10% at window 8; caught an off-by-one), the maximum supported context (1M
  tokens x 8 requests), real-trace quantiles.
- Guard cases: each bad input must be rejected on every rank, including a bad value on rank 0 only;
  calls with another device current must give identical results.
- Run the full suite twice. One intermittent hang (2 ranks idle before the kernel, not reproduced in
  3 later runs) was reported as unresolved, not hidden.
- Pinned linters of the target repo (ruff version, black) before pushing.

## 6. PR description
- Behavior and intent, exact validation command and result, and an explicit unverified scope (other
  archs / TP sizes / page sizes, trap paths not injected, graph replays beyond capacity).
- e2e numbers with their noise floor, and where the gain does not apply (§6 of `07`) with the
  factor that went to zero.
