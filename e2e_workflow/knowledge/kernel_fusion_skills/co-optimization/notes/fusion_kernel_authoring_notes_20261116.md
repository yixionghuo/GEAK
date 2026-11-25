# 怎样一次写出高性能融合算子：整理稿（为写 prompt 做准备）

> 依据：
> - 11-12~16 实验，DSv4.1-Flash c32，MI355X，5 个 candidate（C16、C01、C09、C07、C04）及补充实验；目录 `/kmd2/zhaoan/fusion_opt_exp_c32_20261112`；
> - GEAK main（001c5c7）`perf_knowledge` 里与融合相关的卡片：`optimization/kernel_fusion_strategy.md`、`operators/splitk_streamk_gemm/fusion.md`、`optimization/xcd_l2_locality.md`、`optimization/wave_and_grid_sizing.md`、`operators/*/fusion.md` 等。
>
> 你整理的 GEAK fusion skill 还没合进来，稍后补。

---

## 一、从 5 个 kernel 得到的结论

| ID | 融合内容 | 结果 | 决定胜负的因素 |
|---|---|---|---|
| C16 | 26 个 aten 小 kernel → 1 个 | 融合赢，12.75× | 每个 program 处理一行，行与行之间没有依赖；原链几乎全部是 launch 开销 |
| C01 | sort + MXFP8 quant → 1 个 | 融合赢，1.88× | 每个 quant CTA 自己重算 rank，不等 sort CTA；按 token 划分，每个 token 只量化一次 |
| C09 | split-K partial + reduce → 1 个 | 第一轮输（保留 SPLIT_K=8，跨 XCD 交接）；补充实验赢，比单独优化快 1.16× | 补充实验改为 SPLIT_K=2，两段放在同一个 XCD，交接只走本 XCD 的 L2 |
| C07 | q_norm 并进 wq_b prologue | 输 | 按行的 norm 被 256–512 个 N tile 各重算一遍；持久化复用又牺牲了读权重的并行度 |
| C04 | HCB split-K 点积 + norm/sinkhorn | 输 | 每一行要等 80 段 split 全部写完，只能靠 kernel 内跨 CTA 等待；改成整行一个 CTA 后，并行度不够 |

**一句话**：融合能省的只有"每少一次 launch 约 2 µs，加上很小的中间张量读写"。能不能赢，取决于融合后有没有引入**跨 CTA（尤其是跨 XCD）同步、重复计算或并行度损失**。

**融合前后顺序的实验结论**：C01 两种顺序调完结果相同；C09 靠"先单独优化，再把发现带进融合"才赢。所以**顺序本身不重要，重要的是融合版的搜索空间要覆盖足够多的并行划分方式，并且要用单独优化的发现作为起点。**

---

## 二、核心原则

1. **融合 kernel 的目标**：在每个 shape 上 `t_fused ≤ min(原链, 成员单独优化)`。做不到的 shape 就按 shape 选路，不能为了融合而融合。
2. **先算预算，再动手**：

   融合的收益上限 ≈ (被省掉的 kernel 数 − 0) × 每个 graph 节点约 2 µs + 中间张量读写字节 ÷ 8 TB/s。

   融合的新增代价包括：跨 CTA 同步、重复计算、并行度损失、寄存器合并导致的 occupancy 下降。新增代价超过收益上限的方案直接放弃。
3. **先定依赖类型，再定并行划分，最后才写代码**（见第三节）。依赖类型基本决定了能不能融、该怎么融。
4. **把融合 kernel 写成可搜索的模板**：tile、split 数、划分方式、warps、stages、waves_per_eu、XCD 映射都做成参数，**划分方式本身也是参数**。
5. **用单独优化的发现给搜索空间定起点**：单独优化发现了什么（例如 "split 太多"、"换某个算法更快"），就要把它作为融合版搜索的方向，而不只是搬 tile 参数。

---

## 三、方法：融合前先判断依赖类型

对融合区域里每一对"生产段 → 消费段"，判断消费段的一个工作单元需要读生产段的哪些输出：

| 类型 | 消费段的一个单元需要什么 | 例子 | 怎么融 | 风险 |
|---|---|---|---|---|
| **D0 本地** | 同一行或同一元素，在同一个 CTA 内就能拿到 | C16 metadata；norm+quant；RoPE+写 KV | 直接融，一个 program 处理一行或一个 tile | 低。注意寄存器合并 |
| **D1 可重算** | 需要一个全局量（直方图、rank、offset），但每个 CTA 自己算一遍的代价很低 | C01 sort 的 rank | **用重算代替同步**：每个 CTA 重算自己需要的那部分 | 重算代价要明显小于约 2 µs |
| **D2 小组归约** | 只需要少数几个 CTA 的结果（例如 2–4 段 split） | C09 补充实验 SPLIT_K=2 | 让同一组 CTA **落在同一个 XCD**，最后到达的 CTA 用 cta 级原子 + 本 XCD L2 交接后再做 epilogue | 组内 CTA 会串行；组要小 |
| **D3 全局归约** | 需要大量 CTA 或跨 XCD 的全部结果（整行跨很多 tile，或 80 段 split） | C04；C07 并进上游 epilogue | **一般不融**。或者改成不需要该归约的算法（但并行度可能不够） | 设备级同步 ≥2–3 µs，acq_rel 会触发 L2 写回或失效 |
| **D4 广播或复用** | 消费段的每个 tile 都需要生产段的同一大块输出（整行） | C07 prologue | 只有在重复次数 × 生产段代价远小于约 2 µs 时才融；否则在生产段那边做 epilogue，或者不融 | 重复计算按 tile 数放大 |

判断规则：
- 区域里只有 D0/D1/D2 → 适合融合。
- 出现 D3/D4 → 先尝试换算法或换边界消掉它；消不掉就不融，转做单 kernel 优化。

---

## 四、选并行划分（写代码前）

1. **列出候选划分**，并为每个候选估算 CTA 数、每个 CTA 的工作量、同步类型、重复倍数：
   - 沿用生产段的划分；
   - 沿用消费段的划分；
   - 混合划分：生产段的 CTA 兼做消费段的工作，例如 C01 的 quant CTA 自己重算 rank；
   - 持久化 kernel；
   - split 数在 1/2/4/8 范围内的各种取值。
2. **并行度底线**：MI355X 有 256 个 CU、8 个 XCD。decode 下小 M 的形状要算清 CTA 数够不够铺满一个 wave；不够时用 split 补并行度（C09 在 T<18 时需要 split-K）。
3. **XCD 意识**：硬件按 `pid % 8` 轮询把 CTA 分到各 XCD。需要互相交接的 CTA 要通过 pid 重映射放到同一个 XCD（参见 GEAK `xcd_l2_locality.md`）。同一个 XCD 内的交接成本很低，跨 XCD 很贵。
4. **寄存器预算**：融合 kernel 的寄存器用量取各段中的最大值。目标是不让原来最快那段的 occupancy 下降（C04 用到 142 VGPR 时，HCB 段的 occupancy 减半）。
5. **epilogue 放置**：只在最终归约完成后做 epilogue，每个输出元素只做一次（GEAK `splitk_streamk_gemm/fusion.md`）。

---

## 五、实现模式（成功案例里用到的）

| 模式 | 用在 | 要点 |
|---|---|---|
| 一个 program 处理一行，多个输出 | C16 | 把一串 elementwise、index、gather 合进一行内完成，一次写出多个张量。只有 num_warps 可调 |
| 重算代替同步 | C01 | 每个 CTA 重算直方图、cumsum、rank，只针对自己的 pair，复杂度 O(TOPK·P)。完全没有跨 CTA 等待 |
| 按"去重后的单位"划分 | C01 | 按 token 划分，而不是按 (token, slot) pair：一个 token 只量化一次，scale 写到 6 个目的地址 |
| 小组 split + 同一 XCD 交接 | C09 补充实验 | SPLIT_K=2，同一 tile 的两段映射到同一个 XCD。partial 用 write-through 写回，用 cta 级 relaxed ticket，最后到达的 CTA 做 reduce 和 epilogue。计数器自复位，不另外起清零 kernel |
| 按 shape 选路 | C09 | 在小 T、中 T、大 T 上分别选择融合版或单次 launch 版，用 guard 保护边界 |

## 六、反模式（失败案例里出现的）

- 用设备级 acq_rel 原子做 kernel 内同步：编译成 `buffer_wbl2` 和 `buffer_inv`，单次就要几十 µs（C04 第一版）。
- 保留原来较大的 split 数（8），只把 reduce 挪进 kernel：省掉的 launch 被交接成本抵消（C09 第一轮）。
- 把按行的操作放进按列切的 GEMM 的 prologue：重复计算量放大 N/BN 倍（C07）。
- 为了避免同步，让一个 CTA 包办整行：并行度跌到原来的几十分之一（C04 补充实验）。
- 原样照搬原链里第二个 kernel 的划分：那个划分是建立在"前一个 kernel 已经全部完成"之上的，进了融合 kernel 就要重算或者等待（C01 补充实验的第一版，按 pair 划分）。
- 依据 profiler 里单个小 kernel 的耗时来估算融合收益：profiler 下每个小 kernel 约显示 4 µs，graph 里实际约 2–2.5 µs，会高估一倍。

---

## 七、MI355X 和测量相关的事实（写进 prompt 的"硬件常识"部分）

- **graph replay 下的 launch 开销**：依赖链上每多一个小 kernel，约多 2.0–2.1 µs。实测脚本是 `stage2/scratch/floor.py`。
- **8 个 XCD，每个 XCD 有独立 L2**：cta 级原子和本 XCD 内的交接很便宜；跨 XCD 需要设备级可见性，代价至少 2–3 µs。
- **HBM 约 8 TB/s**：几 KB 到 1 MB 的中间张量，读写一次不到 0.1 µs，不是融合收益的主要来源。
- **计时口径**：graph replay、冷 cache（读驱逐），按负载里每个 shape 的调用次数加权。比较时以整条链的总时间为准，profiler 的单 kernel 时间只用来看占比。
- **精度**：默认要求与线上路径逐位一致。如果改变了求和顺序（split 数、归约树），要写明容差，并和单独优化后的结果放在同一个精度档位比较。

---

## 八、prompt 骨架（初稿，等加入 GEAK skill 后再定稿）

1. **输入**：
   - 融合区域：成员 kernel、seam、shape 和权重（来自 trace 和 workload）；
   - 原链的参考实现；
   - 单独优化的结果，作为底线和方向提示。没有的话，先做 Tier A/B，并限定时间。
2. **第一步，算预算**：收益上限 ≈ 省掉的 launch × 2 µs + 中间张量字节 ÷ 带宽。上限明显小于预期收益的，直接返回"建议不融合"。
3. **第二步，依赖分析**：逐对标出 D0–D4（第三节）。出现 D3/D4 时，先尝试改算法或改边界，消不掉就返回"不融合，转单 kernel 优化"。
4. **第三步，选并行划分**：至少列出 3 个候选（第四节），写出 CTA 数、同步类型、重复倍数、寄存器估算，选出主方案和备选方案。
5. **第四步，写可搜索的模板**：划分方式、split 数、tile、warps、stages、waves_per_eu、XCD 映射都要是参数。同步只允许使用 cta 级或 XCD 本地的方式，禁止设备级 acq_rel。
6. **第五步，搜索**：搜索空间的起点要包含成员的原始配置和单独优化的发现；按 shape 分档；必要时做 Tier C（结构改动），改完再搜一轮。
7. **第六步，验收**：
   - 精度逐位一致，或在声明的容差内；
   - 能被 graph capture；
   - 每个 shape 都比较 `t_fused` 和 `min(原链, 单独优化)`，输的 shape 按 shape 选路回退；
   - 报告四列耗时：原链、单独优化、融合第一版、融合调优后。
8. **输出**：kernel、配置表、选路 guard、单测结果；如果失败，写明属于 D3/D4 中的哪一类、为什么。

---

## 九、和 GEAK 现有知识的差距（待合入你整理的 skill 后核对）

| 方面 | GEAK 现状 | 本次实验补充 |
|---|---|---|
| 何时不融合 | 有"ideal grid/tile 不同时不要融"这类定性描述 | 给出 D0–D4 依赖分类和量化预算（graph 下 launch 约 2 µs） |
| 多 XCD | 有 XCD L2 亲和和 swizzle 的 GEMM 调优卡 | 补充"kernel 内交接必须留在同一个 XCD"，以及设备级 acq_rel 的实测代价 |
| split-K | 有"epilogue 只在最终归约后做"的卡片 | 补充"融合时把 split 数作为搜索维度，小 split + 本 XCD 交接"的成功模式 |
| 单 kernel 优化和融合的关系 | 两条线各自独立 | 先测单独优化作为底线，并用它的发现给融合搜索定起点；每个 shape 取较快的一方 |
| 测量 | — | profiler 下小 kernel 的耗时会被放大，以 graph replay 的整链总时间为准 |
