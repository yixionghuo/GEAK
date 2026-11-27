# fusion-analysis skill 修改建议（适用于 AgentX，跨模型、跨场景）

适用范围：`e2e_workflow/knowledge/kernel_fusion_skills/fusion-analysis/`（SKILL.md、references/），以及 workflow 里直接使用 skill 输出的几处代码（`kfGain` / `estGainPct`、`speedupNeedsReview`、槽位组合、精度 gate）。
依据：GEAK 在 AgentX 上的多次运行（含 GLM-5.2 TP4/DCP4 c40），`co-optimization/references/07_kernel_to_e2e.md`，以及 `refactor/FUSION_STRATEGY_ALGO.md`。
目标：换一个模型、并行方式或并发点，规则仍然成立。下文不写任何具体模型的数字，只写方法；需要的数字都从当次运行的 trace、server log 或已完成的 A/B 中现算。

---

## 0. 现状的问题

现在的估算链是：区域省下的设备时间 × 阶段权重 = e2e %。这条链在 AgentX 下有四处系统性偏差：

| # | 偏差 | 后果 |
|---|---|---|
| P1 | 只算设备时间，**不看目标指标怎么对时间做出响应**。AgentX 是固定并发的闭环回放，评分轴常是 P90 ITL / interactivity、TTFT 这类尾部和排队指标 | 缩短最长 step 的优化（大 prefill chunk、长 verify step）被系统性低估；只改平均时间的优化被高估 |
| P2 | 融合收益上限只有"省 launch + 省中间字节"两项，**没有"通信/计算重叠"这一项** | 通信类区域（TP/EP/DCP/CP 的 collective 两侧）几乎拿不到估算值，排名靠后，很少被选中 |
| P3 | 区域收益里**混着两条路线都能拿到的重算子调优**，融合独有的部分没有单独列出 | 融合候选和同槽的单 kernel 候选被当成竞争关系，其实它们常常能叠加；区域是"赢"了还是只是"重复做了同一件事"，说不清 |
| P4 | 输出只有一个点估计（`est_gain_pct_expected`），排序、A/B 选择和"可疑提升"复测都基于它 | 估计偏低的真实收益会触发复测，复测失败就被丢掉；估计在噪声以下的候选仍然占用整个 A/B 槽位 |

下面的修改按"分析阶段（skill）"和"使用阶段（workflow）"分开写。每一条都只依赖当次运行能拿到的数据。

---

## 1. 原则（写进 SKILL.md 开头，替换或补充现有的 Core principle）

1. **先建指标响应，再估收益。** 收益的单位是目标指标，不是设备时间。对每个阶段、每类 step，先回答"这类 step 变快 x，目标指标变多少"，再去估各个候选。
2. **收益分成两类，分开计算。** 一类是共享收益：两条路线都能拿到，例如重算子本身的调优。另一类是融合独有收益：吸收的轻量 kernel、中间结果不落 HBM、通信或等待被藏进计算、一次读取被多个消费者共用。路由只看融合独有收益。
3. **先确定基线，再谈融合。** 融合版的比较对象，是"每个成员都已经调到最好"的分开版（`t_floor`），而不是原始实现。成员能单独优化的，先单独优化，融合建在它之上。
4. **结论分三种，按 shape 分别下。** 每个 M（或 bucket）给出：融合、不融合、实测后再定。不允许写"倾向融合"。同一个区域在不同 M 上可以走不同路线（host 端按形状分支，要能被 graph 捕获）。
5. **估计带区间，并和噪声比较。** 每个数都给出下界、期望、上界，以及依据。估计和当次运行实测的 A/B 噪声放在一起看：比噪声还小的候选不值得单独占一次 A/B。
6. **用实测反推模型。** 同一次运行里已完成的 A/B（以及 KB 中同部署、同 workload 的记录）是标定数据。估算模型与实测的偏差要回写，供后续候选使用。

---

## 2. SKILL.md 逐节修改

### 2.1 Step 1 Workload characterization：新增"指标响应画像"（解决 P1）

新增一小节 **Metric response**，要求产出一张表：每类 step 一行（prefill/extend chunk 的各个大小档、decode、target verify、draft）。

| 列 | 含义 | 数据来源 |
|---|---|---|
| 设备时间占比 | 这类 step 在回放中的 GPU 时间份额 | workload summary / trace |
| 时长分布 | 这类 step 单次耗时的 p50 / p90 | trace、server log |
| 对 ITL 尾部的贡献 | 解码请求的相邻 token 间隔中，**含有这类 step 的间隔所占的比例**，以及它们在间隔分布尾部（目标分位数附近）的占比 | server log 的 step 时间线（decode 迭代与 prefill chunk 的交错），或 trace 中 step 顺序 |
| 对排队和 TTFT 的贡献 | 这类 step 是否处在新请求进入的关键路径上（prefill / extend 一般在；decode 一般不在） | 调度器 log：running / queue 长度、TTFT 分布 |
| 敏感度系数 s_k | 这类 step 缩短 1% 时，目标指标变化多少（%） | 按下述方法计算 |

敏感度的计算方法要写成可执行的步骤，不能给常数：
- **尾部指标（P90 ITL / interactivity）**：把 log 里的 token 间隔序列取出来，把含有第 k 类 step 的那些间隔按候选给出的时长缩短量进行重放，重新算目标分位数。这是一个离线的 trace 级"what-if"，不需要 GPU。
- **吞吐和 TTFT**：AgentX 的负载由客户端以固定并发给出，吞吐只在有排队、或者固定时长内对话能推进得更远时才会上升（见 `07` §4）。用调度 log 中的 queue 长度、running 数判断是否处在饱和区；处在饱和区时，按"关键路径 step 的缩短量"折算吞吐；不在饱和区时，吞吐的敏感度记为约 0。
- 拿不到 log 时，把 s_k 记为"未知"，并在 report 中列为缺失信息。**不允许默认 s_k = 设备时间占比。**

**为什么通用**：任何开了 chunked prefill、投机解码或混合 batch 的服务，step 时长都是混合分布；尾部指标对最长那类 step 的敏感度一定不等于它的设备时间占比。这张表只依赖 log 和 trace，换模型、换框架都能算。

### 2.2 Step 4 Candidate evaluation：每个候选记录"改变的是哪类 step 的哪段时长"

在现有字段之外新增：
- `step_deltas`：按 step 类别和 M bucket 列出预计缩短的时长（µs 或 %），而不只是区域总体加速比；
- `shared_gain` 和 `fusion_only_gain`：两类收益分开（定义见 2.3）；
- `per_m_route`：每个 M bucket 的结论（融合 / 不融合 / 实测后再定）和依据；
- `composes_with`：可以叠加的单 kernel 候选（比如区域内的重算子已经有独立的 head 候选），以及叠加的前提（需要合并哪些文件或 seam）。

### 2.3 Step 5 Fusion against the single-kernel route：拆开归因，补上重叠项（解决 P2、P3）

把现有的上限公式

```
upper = nodes_removed × ~2 µs + intermediate_bytes ÷ BW
```

改为三项之和，并分别列出依据：

```
fusion_only_upper(per occurrence) =
    absorbed_light_kernels_time          # 被吸收的轻量 kernel 的实测耗时（节点开销 + 自身耗时）
  + intermediate_bytes ÷ BW_stream       # 中间结果不落 HBM
  + overlap_hideable                     # 新增：可被藏住的通信/等待时间
overlap_hideable = min(T_comm_or_wait_on_critical_path, T_independent_compute_available) × η
```

- `T_comm_or_wait_on_critical_path` 必须取**关键路径上实际等待的时间**：collective 取最慢 rank，包括 spin 和等待；不能取各 rank 最小值，也不能取剔除 spin 之后的值。
- `T_independent_compute_available` 是同一 step 里和这段通信没有数据依赖、可以并行的计算（下一个分块、下一层的独立部分、其他流）。
- `η` 是重叠效率，用当次运行或 KB 里同类重叠的实测值；没有实测时给一个区间（例如 0.5–0.8），**不要给单点**，并把这个候选标为"实测后再定"。
- 通信下限要按"所有能并发的链路带宽之和"计算（collective 实现可以多路径并行）。点对点替换 collective 的方案，必须先和这个下限比较。

再加一条归因规则：
- **共享收益（`shared_gain`）**：区域内重算子的独立调优空间，两条路线都能拿到。路由时把它从两边同时扣掉：`route = fusion` 当且仅当 `fusion_only_gain` 在最优分开版之上仍有 ≥ max(5%, 测量噪声) 的区域内收益。
- 如果 `shared_gain` 远大于 `fusion_only_gain`，结论应该是"先走单 kernel 路线，融合作为后续叠加项"，而不是"融合"或"不融合"二选一。

### 2.4 Step 6 Ranking：排序用区间和敏感度（解决 P4）

- 排序键从 `est_gain_pct_expected` 改为三元组：`(e2e 下界, 期望, 上界)`。其中 e2e = Σ_k s_k × step_delta_k，s_k 来自 2.1 的敏感度表。
- 推荐集合里，**至少保留一个"上界高、期望不确定"的候选**进入实测，也就是给探索留一个名额，避免估算模型的系统性偏差永远得不到纠正。
- 期望值低于当次 A/B 噪声底的候选，标为 `batch_only`：不单独占 A/B 槽位，留给后续"多个小改动叠加后一起测"。

### 2.5 Estimation 一节：改写为两层

把现在的单层估算拆成：
1. **Kernel 层**（保留现有内容，并补上 2.3 的重叠项和归因）：区域在每个 M 上的 `t_chain`、`t_floor`、`t_fused` 区间。
2. **Serving 层**（新增）：用 2.1 的敏感度表，把 step 级的缩短折算成目标指标的变化。写清楚以下几点：
   - 尾部指标对"最长 step"的敏感度一般大于它的设备时间占比；平均指标则相反；
   - 吞吐只在饱和区才会响应；
   - 必须给出区间；s_k 未知时只能输出 kernel 层结论，并把候选标为"实测后再定"。

### 2.6 references/heuristics.md：补充下面这些通用先验

| 情况 | 先看什么 | 说明 |
|---|---|---|
| 目标是尾部延迟（P90 ITL 等），且 prefill 与 decode 混合调度 | 最长那类 step 的时长 | 缩短它对尾部的影响常常远大于设备时间占比；按 2.1 的方法算敏感度，不按占比算 |
| collective 夹在两段计算之间（TP/EP/DCP/CP） | 重叠 | 收益主要来自 overlap_hideable，其次才是省 launch；按 shape 门控（大 shape 做流水，小 shape 保持串行） |
| 区域里有一个重算子，并且它已经有独立的优化候选 | 叠加 | 融合应该建在这个重算子的最优实现之上；把它当竞争对手会让两边收益都打折 |
| 改动涉及量化 / 数值路径 | 精度检查的统计功效 | 小样本精度测试的噪声可能比真实差异还大，见 3.4 |

### 2.7 references/report.md：新增三节

- **Metric response**：2.1 的表格，以及每个 s_k 的来源；
- **Attribution**：每个推荐候选的 `shared_gain` / `fusion_only_gain` 拆分，以及 `composes_with`；
- **Calibration**：本次运行中已有的"估计 vs 实测"对照（如果有），以及据此对 s_k 或 η 做的修正。

---

## 3. workflow 侧的配套修改（skill 输出的使用方）

### 3.1 HeadKernel / Milestone 的排序（`kfGain` / `estGainPct`）
- 读取 skill 给出的区间和 s_k，按期望值排序；同时保留"上界最高"的一个候选做探索。
- 单 kernel head 没有 skill 估计时，现在用的是 Amdahl（`kf_single_speedup_prior`）。改为同样走 serving 层折算：用这个 kernel 所在 step 类别的 s_k，而不是用全局设备时间占比。

### 3.2 槽位内的组合（P3 的使用侧）
- 当一个融合区域包含某个 head 的 kernel 时，现在是"同槽竞争，择优保留"。改为：
  1. 先跑 head 的单 kernel lane；
  2. 融合 lane 以 head 的最优实现作为分开版基线（`t_floor`），在其上实现融合独有部分；
  3. 同槽 A/B 比较三个版本："head 胜者"、"融合版（已含 head 胜者）"、以及在资源允许时"融合版（不含 head 胜者）"。
- 只有 `composes_with` 判定为不可叠加（同一 seam、不可合并的改写）时，才退回"二选一"。

### 3.3 "可疑提升"复测（`speedupNeedsReview`）
- 预期上限改为 serving 层上界（Σ s_k × step_delta_k 的上界），不再用 kernel 层的 Amdahl 上限。尾部指标和饱和区的放大效应已经计入上界，真实收益就不会被大面积误判为"可疑"。
- 复测 agent 的超时要按复测本身的 A/B 时长计算：已在分支 `kf-co-opt-mega-20261127-review-timeout` 修复。
- 搜索阶段的 A/B 如果已经跑了 ≥2 个 replica，并且两次结论一致，就可以省掉单独的复测。

### 3.4 精度 gate 的统计功效
- 精度对比要满足：在当前样本量下，两组的差异能和噪声区分开。例如用完整题集，或者做配对比较并报告置信区间。样本量不够时，结论应该是"无法判定"，而不是"回退"。
- 当前小样本 gsm8k 会把 ±2–3% 的抽样噪声判成回退，进而触发不必要的纠错重写。integrator 可以先跑小样本做筛查，但给出 fail 之前必须用足够大的样本确认。

### 3.5 回写标定
- 每次 A/B 完成后，记录 `(候选, 预计的 step_deltas, 预计的 e2e 区间, 实测 e2e)`。
- Report 阶段汇总偏差，按 step 类别给出 s_k 的修正系数，写进 KB（以部署和 workload 为维度）。下次同类部署的分析读取这些系数作为先验。

---

## 4. 验证这些修改的方法（不依赖具体 case）

1. **离线重判**：拿已有运行的 trace、log 和 A/B 结果，用新方法重新估算每个已测候选，比较"估计区间是否覆盖实测"以及排序的变化。覆盖率和排序相关性就是指标。
2. **对照用例**：至少覆盖三类区域：通信夹在计算之间的区域、纯短 kernel 链、重算子加 epilogue；至少覆盖两种评分轴（尾部延迟、吞吐）和两种并发区间（饱和、未饱和）。
3. **回归测试**：skill 的 `kf_trace_digest.py` 增加"step 间隔与 step 类别"的输出，以及敏感度的离线 what-if 计算，并写对应的单元测试。workflow 的估算、复测阈值和组合逻辑在 `scripts/test_*.js` 里补用例。

---

## 5. 改动清单（按优先级）

| 优先级 | 改动 | 位置 | 规模 |
|---|---|---|---|
| P0 | 指标响应画像和 serving 层折算（2.1、2.5） | SKILL.md Step 1、Estimation；`kf_trace_digest.py` 输出 step 时间线 | 中 |
| P0 | 重叠项和共享 / 融合独有归因（2.3） | SKILL.md Step 5、heuristics.md | 小 |
| P0 | 复测阈值改用 serving 层上界（3.3） | e2e_workflow.js `speedupNeedsReview` | 小 |
| P1 | 槽位内组合，融合建在 head 胜者之上（3.2） | e2e_workflow.js HeadKernel 槽位、`kfSlotCand` | 中 |
| P1 | 区间排序和探索名额（2.4、3.1） | SKILL.md Step 6、`kfGain` | 小 |
| P1 | 精度 gate 的统计功效（3.4） | roles/e2e_integrator.md、精度脚本 | 小 |
| P2 | 按 M 分别给出路线（2.2） | SKILL.md Step 4、fusion_ut_contract（cases 按 bucket 给出路线） | 中 |
| P2 | 回写标定（2.7、3.5） | report.md、Report 阶段、KB | 中 |
