# Kernel fusion 协同优化 skill — 中文摘要

## 核心结论：融合优先还是 kernel 优化优先？

两者都不能单独优先。**融合只消除等待，不消除计算。**六个算子的实测都一样：只做融合最多带来约 5%，有三个是负收益；
收益基本都来自融合之后在结构*内部*做的 kernel 优化，或者是融合前就该做的负载均衡。

| 算子（GPU） | baseline | 只做融合 | 最终 | 最大单项收益 |
|---|---|---|---|---|
| MegaMoE EP8 8192 tok/rank（MI355X） | 4.70 ms | 1.045x | ~1.22x | GEMM2 按列分组的 claim 顺序 |
| MegaMoE EP8 64 tok/rank（MI355X） | ~344 µs | 1.05x | 1.18x | 同 XCD 内无 fence 交接 |
| Muse attention 块 TP4（MI355X） | 8 次 launch | 0.685x（grid barrier） | 1.174x | 按 token 均分 attention（不需要融合） |
| Muse MoE TP4/EP4（MI355X） | 7 次 launch | 1.038x | 1.08x（可部署） | EP 负载均衡（要复制全部 expert） |
| Muse attention 输出段 TP8（MI300X） | 8 次 launch | 1.016x | 1.497x | router 通过 norm 做代数分解 |
| Qwen3 MoE EP8，调优基线（MI300X） | 8 次 launch | 0.315x | 1.054x | 每个目标 rank 只发布一次到达 |
| Qwen3 MoE EP8，scratch 基线（MI300X） | 10 次 launch | 0.837x | 1.25x | 权重按 tile 连续排布 |

## 经验汇总（对应 `SKILL.md` §0.1，每条至少在两个算子上成立）

**动手之前**
1. 基线必须是现在线上跑的生产路径，并且开 graph。同一个 Qwen3 融合，对 scratch 基线是 1.25x，对调优过的库 kernel 只有 1.05x。
2. launch 次数不是成本。graph 里的 kernel 本来就是首尾相接；省掉 7 次 launch 只省了约 1 µs（模型预测 18.5 µs），Muse MoE 只减 launch 收益为 0。要按等待、同步和没重叠上的时间来估收益。
3. 覆盖范围按目标配方的 batch 分布来定。e2e 收益跟着覆盖到的 GPU 时间走，而不是跟着 kernel 加速比走；配方一变可能直接变成 0。
4. 先看时间花在哪：生产已经跑到硬件上限的部分（expert GEMM 到了 HBM 的 91%）留在外面，只融合它前后的固定开销。
5. 先做负载均衡再融合。CTA 之间均分（按 token 均分 attention，不融合就 1.22x）和 rank 之间均衡（EP 均衡，MoE 最大的一项）都比任何融合手段收益大。

**写融合 kernel**
6. 第一版骨架低于 1.0x 是常态（六个里 0.32x~1.05x）。先把每个 stage 补回单独跑的水平，再谈重叠。
7. 不用 grid barrier，一开始就用单调计数器做就绪信号（每个 barrier 在 1 CTA/CU 下 4~6 µs）。同步范围尽量窄：rank 内用 agent scope，只有跨卡用 system scope。
8. 不要为了塞进融合 kernel 而用别的 DSL 重写调优过的 stage（慢 5~10%，或者只有 HBM 的 13~19%）。复用原实现，或者留在外面。
9. 按 1 wave/SIMD、256 寄存器设计。所有 2-wave 尝试都放不下或更慢。常驻是正确性前提：依赖的工作用 ticket 认领，不能靠 program_id 顺序（会死锁）。
10. 预取只读本来就要读的数据，并且放在本来就要等的地方。额外提前读权重在四个算子上都是负收益。
11. 注意 vmcnt 是按顺序等待的：预取或推送放错位置会和本想重叠的 load 一起被等掉；条件 load 可能让编译器每个块插一次 vmcnt(0)。
12. 找能把计算挪出关键路径的代数变换（router 通过 norm 分解，+18 pt）；让归约搭上本来就有的通信（split-K 放进 reduce-scatter）。
13. 跨卡通信：数据本身当就绪标志、push 不用 pull、轮换目标卡、每个目标只发布一次、写对端内存不要每次 store 都等。

**测量和上线**
14. 分辨率：UT 每对约 2~3%，serving 吞吐 ±3%、P90 ±1.5%。连续 2~3 轮都低于分辨率就停止微调，改结构或换瓶颈。
15. 用 serving 探针确认：按 batch 记录每次 forward 的 GPU 时间，和"UT 加速比 × batch 分布"的预测对比。说没收益时要指出是哪个因子变成了 0。
16. kernel 变快会改变时序，暴露原本就有的分布式 bug（TP 不同步变成死锁）。要检查各卡一致性，两边都打同样的修复。
17. 做成库算子：host 检查只依赖各 rank 一致的值；初始化和释放是集合操作；kernel 内等待有上限并记录错误；UT 覆盖所有输出和坏输入。graph 下 host 检查只在 capture 时跑一次。

## 融合到一半时怎么优化（`09` + `tools/`）

骨架正确但不够快时，每轮按这个循环做：
1. **按段打时间戳**：HIP kernel 引入 `tools/region_stamp.h`，每段结束处加 `RS_MARK`（`-DREGION_STAMP=1` 时才编进去），
   再用 `tools/region_timeline.py` 输出每段结束时间的中位数和最大值、不均衡量，以及沿"最后到达者"串起来的关键路径。
   重放要连续跑，不要每次同步，否则跨卡等待会被放大 4~5 倍。CTA 之间小于约 2 µs 的差异没有意义（本机实测跨 XCD 时钟偏差上限 1.25~1.73 µs）。
2. **先算每段下限，再动手**（`09` §3 工作表）：要读写的字节 ÷ 带宽、计算量 ÷ 实际算力、串行步数 × 延迟，
   得到效率和可省空间。本机 MI355X 实测：HBM 流式读 7.0~7.1 TB/s（达到约 6.2 TB/s 即视为贴上限）、
   BF16 GEMM 1.27 PFLOP/s、FP8 GEMM 2.66 PFLOP/s、单 peer 拷贝 60 GB/s。效率 ≥ 0.85~0.9 的段只有减字节才有用。
3. **按现象查对照表**（`09` §4，S1~S13）：每行给出怎么看到这个现象、先查什么、`02` 里的手段编号、`04` 里不要再试的编号
   （L 是失败做法，T 是结构性陷阱，H 是 hang 排查项）。前六行对应最常见的情况：段比单独跑慢、等生产者就绪时间长、
   等跨卡数据时间长、有 spill、尾部有空闲 CTA、已贴 HBM 上限。
4. 一次只加一个手段，做成编译开关；**看 ISA 和寄存器信息**（`09` §5，命令都在本机验证过）：HIP 用
   `-Rpass-analysis=kernel-resource-usage`、`--genco -save-temps`、`clang-offload-bundler` 拆包、`llvm-readobj --notes`、
   `llvm-objdump --offloading`（从 .so/可执行文件里取 code object）；Triton 用 `c.asm` / `c.n_regs` / `c.n_spills`，
   或者 cache 目录里的 `.amdgcn`（`AMDGCN_ENABLE_DUMP=1` 输出到 stdout，且只在真正编译时才打印）。
   另有 occupancy 计算公式、热循环里该查哪些指令，以及怎么确认"关掉的分支"没有改变寄存器分配。然后做配对 A/B 并重新打时间戳。
5. 连续 2~3 轮低于分辨率就停，换结构、减字节或收手。

每轮"看到什么数据 → 怎么判断 → 选了什么方向"的完整记录：Muse attention 在 `09` §6，MegaMoE 8192 和 64 在 `05` §A.1、§B.1。
最大的几步都来自测量本身：跨卡时间是测量假象、2 倍重复读取、ISA 里隐藏的 vmcnt(0)、"融合后 GEMM2 时间等于单独跑"说明丢了权重复用。
attention 类手段在 `02` §G（新增 G11 最后到达的 CTA 做合并、G12 能一轮合并就一轮、G3 写清了不带 running max 的 softmax 的数值余量检查），
norm / epilogue 类手段在新增的 `02` §N。

**融合成一个 kernel 之后才能做的优化**（不是把各 stage 简单串起来）：`02` §0 有一张索引表，按五类机制排好，
每条附编号、前提条件和实测收益：
1. 提前做下一个 stage 不依赖上游的部分（A1、A3、A6、G9）；
2. 把工作挪到有空闲的地方（A10 代数拆分、A11 归约搭在通信里、G8、B7、B6）；
3. stage 之间交接比 kernel 边界便宜（C9、C2、C3、C5、C6、G11）；
4. 按 CTA 流水而不是整网格串行（B9、A9、B3、B4、B5、B10）；
5. 跨 stage 共享片上资源（E3）。
前提是各 stage 先补回单独跑的水平（B1、E1、E2、F）。证据只来自 gfx950 和 gfx942。

## kernel 收益到 e2e 收益（`07`）

kernel 加速 × 块在一步里的占比 × 覆盖的 batch 占比 × 这类 forward 占 GPU 时间的比例 = GPU 时间节省；服务器是瓶颈时才变成吞吐，否则体现在交互性。

- Muse Spark 1027 镜像：verify 一步在 batch 3~8 快 6~16%，c16 吞吐 +7.9%、P90 交互性 +15.2%；c36 覆盖只有 5%，没有收益。预测和实测吻合（c16 verify 节省预测 9.4%，实测 7.5%）。
- 1106 镜像同样的 kernel 测不出收益：c36/c40 batch 变大超出覆盖；低并发点改成 TP8 或被删；生产路径变快，c28 每步加速从 6~16% 降到 3~11%，估算 e2e 只剩 2~3%，落在噪声内。

## 推荐顺序

1. 定基线（生产路径）和覆盖范围（配方的 batch 分布）；先做负载均衡、去掉多余计算。
2. 各 stage 单独调好、量好，作为 parity 基准。
3. 逐条边判断要不要融合：能否部分交接、消费者资源是否空闲、生产者 wave 数是否大于 1；不满足但能解锁 kernel 级手段的也可以融合。
4. 先搭只求正确的骨架：就绪计数器、不用 barrier、带上每个 stage 的调优配置。
5. 补回 parity，再加只有融合才能做的手段，每条都做配对 A/B。
6. 按 bucket 针对新结构重调常数。
7. 归因：同一棵树开关 fusion 做 A/B；再用 serving 探针确认 e2e。

## 目录结构

```
kernel_fusion_skill/
├── README.md                          英文说明
├── README_zh.md                       本文件
├── SKILL.md                           入口：结论、经验汇总（§0.1，18 条）、流程、手段总表、硬规则
├── references/
│   ├── 01_decide_and_order.md         融合判定、基线和覆盖、更便宜的替代手段、顺序、regime
│   ├── 02_in_fusion_levers.md         融合内手段（A overlap / B 调度 / C 同步 / D 队列 / E 配置 / F codegen /
│   │                                    G attention 与小 GEMM / N norm 与 epilogue，六个算子的证据合在一起）
│   ├── 03_measurement_and_attribution.md  profiling、stage meter、overlap meter、A/B 与归因
│   ├── 04_rejected_and_pitfalls.md    实测失败的做法 L1~L57、结构性陷阱 T1~T15、hang/fault 排查 H1~H12
│   ├── 05_case_studies.md             各算子的逐轮收益表（证据），MegaMoE 附每一步的判断过程
│   ├── 06_plan_template.md            计划和报告模板
│   ├── 07_kernel_to_e2e.md            kernel 收益到 e2e：覆盖率、预测、噪声、配方变化、serving 约束
│   ├── 08_shipping_a_fused_kernel.md  做成库算子：host 检查、集合式初始化/释放、有界等待、UT、PR
│   └── 09_mid_fusion_diagnosis.md     融合到一半时怎么优化：段时间线和关键路径、下限工作表（含 MI355X 实测校准值）、
│                                        现象 S1~S13 → 手段 → 别再试对照表、ISA/寄存器查看命令、occupancy 公式、完整例子
└── tools/
    ├── region_stamp.h                 HIP kernel 按段打结束时间戳（-DREGION_STAMP=1 才编进去）
    ├── region_timeline.py             汇总：每段表格、不均衡、关键路径
    └── examples/
        ├── stamp_demo.hip             可运行的示例 + 跨 XCD 时钟偏差测量
        └── hbm_read_bw.hip            测 HBM 实际读带宽，用于算下限
```

`02` 和 `04` 的条目带两类标记：可移植性标签 `[AMD]`、`[gfx950]`、`[FlyDSL]`、`[multi-rank]`；证据来源 `{attn}`、`{moe}`、`{attn-out}`、`{qwen}`（不标的是 MegaMoE）。换硬件、DSL 或 regime 时保留机制、重测数值。

数据来源：GEAK `perf_knowledge/expert_skills` 和 2026-09 至 11 月的 mega 运行记录；/sgl-workspace/ref 里 Muse Spark、Qwen3 的 mega 运行包，Muse 1027 e2e 数据，aiter PR #7870 的评审结论。数据都是 MI355X/MI300X 上测的。
