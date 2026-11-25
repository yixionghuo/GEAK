# 生成高性能融合算子（prompt v2）

> 在 GEAK 里的用法：KernelFusion 打开时，`e2e_workflow.js` 把一个 FUSION REGION 任务交给 kernel lane（`kernel_lane.js`），lane 先读本文件和同目录的 `SKILL.md`，再读 `../fusion-kernel/SKILL.md`。原来的 `{{...}}` 输入槽由 GEAK 交给 lane 的内容填充（见下面"输入"），不需要再替换。一次只处理一个融合区域；多个区域由 GEAK 并行。
> 依据：GEAK mode=mega 的 kernel fusion skill（6 个算子的实测经验），以及我们在 DeepSeek-V4.1-Flash c32（MI355X）上做的 5 个 decode 小算子融合实验。
> agent 可读的参考资料：本目录（`e2e_workflow/knowledge/kernel_fusion_skills/co-optimization/`：SKILL.md、references/01–09、tools/），以及 GEAK `perf_knowledge/`。下文的 "skill `NN`" 指本目录 `references/NN_*.md`。

---

## 任务

为下面这个融合区域写一个**高性能**的融合 kernel。

**完成标准**：在每个目标 shape 上，融合 kernel 的耗时都不超过 `t_floor`，其中 `t_floor = min(原链, 成员各自单独优化后)`。只是写出一个能跑、结果正确的融合 kernel，不算完成。

**写之前必须先建立的认识**：
- 融合省掉的只是 kernel 之间的等待，计算量一点都不会少。在 CUDA/HIP graph 里，每少一次 launch 只省约 2 µs。
- 第一版融合骨架通常比原链还慢（实测 0.32×–1.05×）。真正的收益来自在融合结构内部做的 kernel 级优化，这部分在已测案例里占全部收益的 67%–100%。
- 所以要**先建模、先设计**，从写第一行代码开始就把并行划分、同步范围、寄存器和各段的调优配置一起考虑进去。

## 输入

GEAK 交给 lane 的内容对应原 prompt 的各个输入槽：

- 融合区域：lane 任务文本里的 region JSON（`FUSION REGION <id> (<level>)` 之后）：`kernels_in_order`（成员 kernel 及调用顺序）、`seam`（要替换的 sglang/aiter 函数，`target_callable` + `source_file`）、`phases`、`m_range`、`inputs_outputs`、`launches_before/after`、`suggested_design`。
- 参考实现：区域 UT 的 baseline leg（`baseline_overlay/`，线上 serving 栈的冻结快照）和 `baseline_ref/*.orig`（原生源码，只读）。它就是生产中正在服务的路径；不要另搭参考链。
- shape 和权重：区域 UT 的 `meta.json` 里的 `cases[]`（`sig`、`step_type`、`width`、`context`、`weight`，weight 是线上占比）和 `regime.speculative`；它们来自本次运行的负载分布。分析依据在 `analysis.json`。
- 已有的单独优化结果（如有）：任务文本或 `analysis.json` 里给出的成员单独优化结果（例如同一热点 HeadKernel 单 kernel 路线的结果、`single_kernel_gain_pct_expected`）。没有就按第 0 步自己做。
- 环境：lane 的 `gpu_ids`（多 rank 区域另有 `gpus_per_job`），任务目录里唯一可写的是 `kernel_src/`；框架源码只读。

## 硬件和测量常识（MI355X / gfx950，其他硬件需要重测）

- 256 个 CU，8 个 XCD，每个 XCD 有自己的 L2。CTA 按 `pid % 8` 轮询分到各 XCD。每个 CU 的 LDS 为 160 KB。
- HBM 实测流式读约 **7.0 TB/s**，达到 6.2 TB/s 以上即视为到顶。FP8 GEMM 实测约 2.66 PFLOP/s；瘦长 shape 的实测值会低得多。
- 同一 XCD 内交接很便宜：agent 原子 + 消费端只失效 L1。跨 XCD 的 agent/system release 会编译成 `buffer_wbl2`/`buffer_inv`，单次就要 2–3 µs 起，按 CTA 做时是几十 µs。
- graph replay 下，依赖链上每个小 kernel 约 2 µs；profiler 会把它放大到约 4 µs。**比较快慢一律以 graph replay 的整链时间为准**。成员时间相加不等于整链时间。
- 计时口径：graph replay，冷 cache（每次前读驱逐至少 512 MB），按调用次数加权。测量分辨率约 2–3%。

## 工作流程（每一步都要产出对应文件，不能跳过）

### 第 0 步：基线和对等底线 → `baseline.json`
1. 测原链。基线是生产路径（已调优的库 kernel，开 graph），不能用自己随手搭的链。
2. 如果没有单独优化结果：不融合，对每个成员做 Tier A（换后端）和 Tier B（调配置，按 shape 分桶），限时约 1 小时。记录每个成员在每个 bucket 下的调优配置和耗时，这就是**对等底线**。融合 kernel 里的每一段都必须恢复到这个水平。
3. `t_floor[shape] = min(原链, 单独优化)`。
4. 记录单独优化中**影响结构的发现**（例如"split 太多是负担"、"某个算法更快"、"该走单次 launch"）。第 6 步搜索时必须以这些发现为起点。

### 第 1 步：预算和下限 → `design.md` §1
- 融合收益上限 ≈ (原链节点数 − 融合后节点数) × 2 µs + 中间张量字节 ÷ 7 TB/s。
- 按 skill `09` §3 的下限表，算出每一段的必需字节、FLOP、串行跳数，得到下限。效率 = 下限 ÷ 实测。效率 ≥ 0.85 的段只能靠减少字节来提速。
- 如果上限明显不够抵消预计新增的代价（同步、重复计算、并行度损失），在报告里写明，并直接考虑按 shape 选路回退或不融合。

### 第 2 步：逐条边做依赖分析 → `design.md` §2
对每条"生产段 → 消费段"的边，填写 skill `06` 的 edge 模板，另外给出依赖类型：

| 类型 | 含义 | 默认做法 |
|---|---|---|
| D0 本地 | 同一行或同一 tile，在同一个 CTA 内 | 直接融；一次读行，把行内操作全部串完（N2） |
| D1 可重算 | 需要一个全局量，但每个 CTA 自己重算很便宜 | **用重算代替同步**，没有任何跨 CTA 等待 |
| D2 小组规约 | 只依赖少数几个 CTA（split 数为 2–4） | 同组 CTA 放在同一个 XCD（B4），由最后到达者合并（G11），只合并一轮（G12） |
| D3 全局规约 | 依赖大量 CTA，或需要跨 XCD | 默认不融，先尝试下面的改写 |
| D4 广播/复用 | 消费段的每个 tile 都要生产段的同一整块输出 | 只有"重复倍数 × 生产段代价"远小于 2 µs 时才融 |

同时回答 skill `01` §2 的三个条件：能否部分交接、生产段运行时消费段的引擎是否空闲、生产段的 wave 数。还要说明"即使不满足这三条，仍然值得融"的 kernel 级理由：提前加载消费段的独立操作数、在片上或同一 XCD 内交接、填尾部 wave、调整 claim 顺序获得复用。**如果融合的唯一理由是"少一次 launch"，就判定为不成立。**

遇到 D3/D4 时，**必须先尝试改写，并逐一记录是否可行**：
- 换算法（例如去掉 split-K）；
- 减少 split 数，并放在同一个 XCD；
- 让最后到达的 CTA 承担消费段的工作；
- 整行 norm 按 N1 的顺序选择归属：一个 tile 覆盖整行 / 发布部分平方和 + 最后到达者 / 按行 × 列块切分；
- 代数拆分，把只依赖早期数据的项提前计算（A10）；
- 换融合边界，例如把 quant 放进生产段的 epilogue（N4），或者融进上游或下游。

全部改写都失败的，结论为"不融，按 shape 选路回退"。

### 第 3 步：设计空间 → `design.md` §3
至少列出 4 个并行划分候选，必须包含：
- 沿用生产段的划分；
- 沿用消费段的划分；
- 重算代替同步（适用时）；
- 小 split + 同 XCD 交接，split ∈ {1,2,4,8}；或者持久化/最后到达者方案；
- 按 shape 选路（不同档位用不同方案，或者回退到不融合）。

**不要照搬原链里第二个 kernel 的划分方式**：它的前提是前一个 kernel 已经全部完成，放进融合 kernel 后会带来重算或等待。要按"去重后的单位"重新划分，例如按 token 而不是按 (token, slot)。

每个候选给出：CTA 数和 wave 数（对照 256 个 CU、8 个 XCD）、同步类型和范围、预计同步代价、重复计算倍数、预计 VGPR/AGPR/LDS（按 1 wave/SIMD、256 寄存器设计；LDS 取各段生命周期内的最大值，不是相加）、预计耗时、取舍理由。选出主方案和至少一个备选。

### 第 4 步：写骨架（可搜索的模板）
- **复用各成员已调优的计算主体**，只改归属、放置、排队和发布方式。不要为了塞进融合 kernel 用另一种 DSL 重写已调优的库 kernel；做不到复用的那一段留在 kernel 外。
- 每一段的调优配置通过 **bucket selector** 接入：常数按 shape 档位在 selector 里推导；只有结构开关用 env。
- 划分方式、split 数、BLOCK_*、num_warps、num_stages、waves_per_eu、claim 顺序、XCD 映射全部参数化，**划分方式本身也是参数**。
- 同步规则：
  - 不用 grid barrier，改用就绪计数器或最后到达者；
  - 同步范围尽可能窄，同一 XCD 内不加 release fence；
  - write-through 只用在需要跨 die 的数据上；
  - 计数器单调递增、独占 64 B 一行、自复位；
  - 被其他 block 写的 buffer 不加 `const`/`__restrict__`；
  - epilogue 只在最终规约完成后做，每个输出只做一次；
  - 原子操作不要放在热循环里。
- 写完立刻读 metadata 和 ISA（skill `09` §5）：VGPR/AGPR/SGPR、spill、LDS、occupancy，热循环里有没有 `vmcnt(0)` 或 `readfirstlane`。

### 第 5 步：恢复对等
- 用区域时间线（`tools/region_stamp.h` 和 `region_timeline.py`，HIP 适用）或下限探针，测每一段在融合 kernel 内的耗时。探针举例：去掉等待（结果会错，只用来计时）、只跑生产段、去掉 epilogue。
- **任何一段在融合内比它单独调优后慢，都是缺陷，必须先修**，修完再做第 6 步。常见原因：没有带上调优配置；claim 顺序破坏了复用；寄存器合并后 occupancy 下降；codegen 问题（spill、嵌套循环、条件 load 引入的 vmcnt）。

### 第 6 步：融合专属 lever + 重调常数（循环执行）
每轮按 skill `09` §1：时间线 → 每段下限 → 选关键路径上"实测减下限"最大的段 → 查症状表 S1–S13 → 只改一个 lever，用编译期开关 → 成对 A/B → 记录（含"不要再试"的项）。

优先考虑的 lever：
- A1/A6：等待生产段时，先发射权重、scale、描述符的加载和初始化；
- G11/G12：最后到达者一轮合并；
- C2/B4：同 XCD 交接；
- N2/N4：行内操作串联，quant 放进 epilogue；
- A10：代数拆分；
- B6/B11：均衡尾部、切分不规则的工作；
- F1–F12：codegen。

`04` 里已经验证会输的做法，没有新理由不要再试：额外的提前读、跨 XCD 偷任务、grid barrier、把"一次 launch"当目标、加深预取或流水（某段已经到下限时）、用另一种 DSL 重写已调优的段等。

**每个结构 lever 都要在常数重调后再评估**。搜索起点必须包含：各成员的原始配置、第 0 步记录的结构性发现、第 3 步的所有候选划分。按 bucket 扫描，相互耦合的常数一起扫（例如 claim 顺序和 chunk 大小）。

停止条件（满足任一，并且要有证据）：
- 每个 shape 都达到 `t_fused ≤ t_floor`；
- 连续 2–3 轮收益低于约 2–3%；
- 下限探针表明已经到底，例如"去掉等待后"的耗时已经大于 `t_floor`；
- D3/D4 的所有改写都试过且失败。

### 第 7 步：验收和归因
- 精度：和生产路径相比，覆盖全部 shape，至少 3 个 seed，2 次 replay，要求逐位一致。如果改变了规约顺序，写明容差，并说明单独优化路径是否也在同一精度档位。同一输入重复跑必须逐位相同，不相同说明有 bug。
- 能 graph capture 和 replay；guard 范围外回退原路径；选路条件只依赖调用时的量。
- 每个 shape 报五列：原链、单独优化、底线、融合第一版、融合最终版。融合输给底线的 shape 自动选路到底线方案。
- 归因：同一份代码分别关闭和打开融合开关（fusion-off / fusion-on），交替跑 A/B，区分融合本身的收益和 kernel 优化带来的收益。

## 输出

GEAK 的产物格式（回收结果目录，`e2e_workflow/scripts/fusion_handback.py` 从 `kernel_src/_handback/` 组装）：

- 代码：`kernel.*`（模板、selector、配置表）、`bench.py` 等全部放在 `kernel_src/` 里（唯一允许修改的目录）。
- `kernel_src/_handback/baseline.json`：第 0 步的基线和对等底线（每个 shape 的原链、单独优化、`t_floor`，以及结构性发现）。
- `kernel_src/_handback/design.md`：预算、逐边分析、设计空间、每轮记录、探针结果（第 1–6 步）。
- `kernel_src/_handback/ut_result.json`：用本区域 UT 跑出的结果，至少包含 `correctness`（"pass"）、`weighted_speedup`；每个 shape 的五列耗时 `shapes: {<sig>: {chain_us, single_opt_us, floor_us, fused_v1_us, fused_final_us}}`；加权汇总；选路表；精度结论；e2e 上限（融合和底线各一份）；VGPR/LDS/occupancy。
- `kernel_src/_handback/notes.md`：结论（只能是以下三种之一：全部达标 / 部分达标、其余选路回退 / 融合不可行。选"不可行"时，必须附上停止条件对应的证据），以及交付的方案和归因。
- `kernel_src/_handback/support.json`：引擎生效范围；按 shape 选路回退（融合输给底线的 shape）逐条写进 `refusals`。
- 结论为"融合不可行"时，`kernel_src/` 交付底线方案（单独优化后的成员，或原样），不要交付更慢的融合。
