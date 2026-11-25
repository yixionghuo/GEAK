# 06 — Plan and report template

## Per edge (Analyze)
```
edge: <producer tile> -> <consumer tile>
  scope needed:          register | LDS/WG | L2 same XCD | cross-XCD | cross-rank
  enforced today by:     launch_boundary | barrier | fence/flag | none_needed
  (1) partial handover:  yes/no + why (consumer's first op on the output)
  (2) consumer engines idle during producer: <pipe utilization, source counter>
  (3) producer waves:    ceil(blocks/(CUs*per_CU)) = N ; window <= (N-1)/N * producer_us
  per-edge sync cost:    <scope, count, cost>
  producer-independent operands of the consumer: [weights, scales, descriptors, ...]
  kernel-level reason to fuse anyway: prefetch | on-chip/XCD-local handoff | tail fill | reuse order | none
  pairing:               whole | tile | tile_cover | tile_reduce  (counter count)
  verdict:               PAYS | DOES NOT PAY | UNKNOWN(<measurement that settles it>)
```

## Per stage parity sheet (before and after fusion)
```
stage | standalone tuned config | standalone us | fused busy us (meter) | fused wait us | VGPR/AGPR/LDS/spill | WG/CU | delta reason
```
A stage with fused busy > standalone is fixed before any overlap lever.

## Per lever (Plan)
```
lever:            <name>  family: overlap|scheduling|sync|config|codegen  tag: fusion-only|restore|shared
structural or constant:  (constants go into the bucket selector)
pipe it fills / work it removes:  <pipe>, current utilization <x%>
headroom bound:   min(pipe headroom, e2e - critical_path) = <us>
expected next critical path:  <node>, <us>
preconditions:    (e.g. same-XCD layout, skew predicate, residency, exact-size descriptors)
coupled with:     <other levers that must land together, or none>
enabling or terminal step:  (enabling = judged on function; terminal = judged on speed vs pre-chain baseline)
guards affected:  <bucket list>, fallback rule for others
rejected-ledger check:  <item in 04 or "new">
```

## Per result (Report)
```
arms (same lease, interleaved): frozen T0 | tree fusion-off T_off | tree fusion-on T_on
kernel-efficiency gain = T0/T_off ; fusion gain = T_off/T_on ; total = T0/T_on
per guard: ratio, per-pair spread, path marker per rank, launches per call
span (not Σ stages); stage meter busy/wait per stage (meter-off timing)
overlap (only if claimed): edge pairs only, cu_fraction, join spread, controls, skew, overhead
correctness: relL2 per size, repeat bit-exact, replay with input change (first vs repeat), ragged, switch test
blocker class if failed: measurement | correctness | liveness | speed | flow
```
