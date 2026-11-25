# kernel_fusion_skill

How to make a kernel fusion faster than the sum of its parts (1+1>2): the order in which to fuse
and optimize, the kernel-level levers that only exist inside a fused kernel, and how to attribute
each gain. Distilled from the GEAK mode=mega MegaMoE EP8 work on gfx950 (8192 tok/rank ~1.22x,
64 tok/rank 1.18x), cross-checked on five more mega fusions (Muse Spark attention block and MoE on MI355X,
Muse attention-output and Qwen3 MoE on MI300X). It also covers how a kernel gain does or does not reach serving
(Muse Spark InferenceX AgentX, images 1027 and 1106), and how to ship a persistent cross-rank op.
Chinese summary: `README_zh.md`.

## Scope

Not everything transfers. Read `SKILL.md` §0.2 first: it says which sections apply to which fusion
type, and what the portability tags in `02` / `04` mean.

```
fusion type
├── epilogue / elementwise (GEMM+act/quant, norm+quant) ─► method + config/codegen (02 §E, §F)
├── tile-internal pipeline (flash attention, fused MLP) ─► 01, 02 §A §E §F, 03
├── multi-stage persistent, single GPU ─────────────────► everything except [multi-rank]
└── cross-rank compute + communication (MoE, GEMM+AR) ──► everything

tags: [AMD] Instinct ISA/topology · [gfx950] lowering/bits/limits/numbers · [FlyDSL] · [multi-rank]
```

## Layout

```
kernel_fusion_skill/
├── README.md                          this file
├── README_zh.md                       Chinese summary
├── SKILL.md                           entry point: answer, consolidated lessons, procedure, levers, rules
├── references/
│   ├── 01_decide_and_order.md         fuse-or-not test, cheaper-lever ladder, ordering, regimes
│   ├── 02_in_fusion_levers.md         lever catalog (mechanism, preconditions, measured numbers)
│   │                                    A overlap/prefetch · B scheduling/locality · C sync scope
│   │                                    D queue structure  · E config/resources  · F codegen
│   │                                    G attention/small GEMMs · N norm/epilogue
│   ├── 03_measurement_and_attribution.md  profiling, stage meter, overlap meter, A/B, attribution
│   ├── 04_rejected_and_pitfalls.md    measured losses (L1-L57), structural traps (T1-T15),
│   │                                    hang/fault checklist (H1-H12)
│   ├── 05_case_studies.md             per-operator gain ledgers (evidence): MegaMoE (with the
│   │                                    reasoning of every step), Muse attn / MoE / attn-output,
│   │                                    Qwen3 MoE (two baselines), workflow lessons
│   ├── 06_plan_template.md            per-edge / per-stage / per-lever / report templates
│   ├── 07_kernel_to_e2e.md            coverage, UT x batch-distribution prediction, noise,
│   │                                    recipe drift (1027 -> 1106), serving constraints
│   ├── 08_shipping_a_fused_kernel.md  host checks, collective setup/close, bounded spins, UT, PR
│   └── 09_mid_fusion_diagnosis.md     mid-fusion loop: region timeline + critical path, floor
│                                        worksheet (MI355X calibration), symptom S1-S13 -> lever ->
│                                        do-not-retry table, ISA/metadata commands (HIP, Triton),
│                                        occupancy formula, worked example, round log
└── tools/
    ├── region_stamp.h                 per-region end stamps in a HIP kernel (-DREGION_STAMP=1)
    ├── region_timeline.py             reducer: region table, imbalance, critical path
    └── examples/
        ├── stamp_demo.hip             runnable demo + cross-XCD clock-skew probe
        └── hbm_read_bw.hip            practical HBM read bandwidth for floors
```

## Reading order

```
SKILL.md
├── deciding whether and what to fuse ─────────► references/01_decide_and_order.md
├── choosing and implementing levers ──────────► references/02_in_fusion_levers.md
│                                                 └─ check first: references/04_rejected_and_pitfalls.md
├── fused but slow: what to do next ───────────► references/09_mid_fusion_diagnosis.md
│                                                 ├─ timeline: tools/region_stamp.h + region_timeline.py
│                                                 ├─ floors: 09 §3 (+ tools/examples/hbm_read_bw.hip)
│                                                 └─ symptom table: 09 §4 -> 02 ids -> 04 L/T/H ids
├── measuring, A/B, attribution ───────────────► references/03_measurement_and_attribution.md
├── calibrating expectations ──────────────────► references/05_case_studies.md
├── writing the plan / report ─────────────────► references/06_plan_template.md
├── will it show up e2e? ──────────────────────► references/07_kernel_to_e2e.md
└── shipping it as a library op ───────────────► references/08_shipping_a_fused_kernel.md
```

## Procedure

```
Step 1  Tune and measure each stage separately              -> parity bar per stage/bucket
   │      (waves/tail, pipe utilization, gap, no-payload control, critical path)
   ▼
Step 2  Per edge: partial handover? idle engine? producer waves > 1?
   │      ├─ all yes ─────────────────────────────► fine-grained edge candidate
   │      ├─ no, but enables a kernel-level lever ─► fuse for that lever
   │      │    (prefetch / XCD-local handoff / tail fill / reuse order)
   │      └─ no ──────────────────────────────────► keep `whole` or leave outside the kernel
   ▼
Step 3  Fused skeleton for correctness
   │      unproven edges = whole, tuned configs via bucket selector,
   │      residency asserted, monotone counters, path marker
   ▼
Step 4  Restore parity: fused busy (stage meter) <= standalone tuned time
   │      claim order, chunk, LDS by lifetime, tile widening, codegen
   ▼
Step 5  Fusion-only levers (paired A/B each)
   │      cross-stage prefetch, fence-free same-XCD handoff, tail fill,
   │      third queue, skew-gated peek, deferred publication
   ▼
Step 6  Re-tune constants per bucket for the new structure
   ▼
Step 7  Attribute: frozen | tree fusion-off | tree fusion-on, span, rank-max, all guards
```

## Where the gain came from (MegaMoE EP8, gfx950)

```
8192 tok/rank   4.70 ms ─ fusion ─► 4.50 ms ─ claim order ─► ~4.06 ms ─ codegen + sync + fill ─► ~3.84 ms
                          (1.045x)              (~1.155x)                                       (~1.22x)
64 tok/rank     344 µs ── fusion ─► 326.5 µs ─ sync scope ─► ... ─ overlap / tail / front end ─► 291 µs
                          (1.05x)                                                               (1.18x)
```

## Consolidated lessons

`SKILL.md` §0.1 lists 18 lessons that held on at least two of the six operators (MegaMoE x2 regimes,
Muse attention block, Muse MoE, Muse attention-output, Qwen3 MoE x2 baselines), grouped as
before writing the kernel / building it / measuring and shipping. `02` and `04` merge the evidence
of all operators by topic, marked `{attn}` `{moe}` `{attn-out}` `{qwen}`; `05` keeps the per-operator
ledgers as evidence only.

```
operator (GPU)                        fusion alone   best        largest lever
MegaMoE EP8 8192 / 64 (MI355X)        1.045x/1.05x   1.22x/1.18x claim order / same-XCD handoff
Muse attn block TP4 (MI355X)          0.685x         1.174x      token-balanced attention (no fusion)
Muse MoE TP4/EP4 (MI355X)             1.038x         1.08x       EP balance (needs replication)
Muse attn-output TP8 (MI300X)         1.016x         1.497x      router algebra through the norms
Qwen3 MoE EP8, tuned base (MI300X)    0.315x         1.054x      once-per-destination publish
Qwen3 MoE EP8, scratch base (MI300X)  0.837x         1.25x       tile-contiguous weight streams
```
