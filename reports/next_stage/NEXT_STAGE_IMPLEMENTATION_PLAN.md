# RFFI 下一阶段执行计划

日期：2026-08-28  
依据：`RFFI_NEXT_STAGE_CODEX_EXECUTION_SPEC_2026-08-28.md`  
状态：根据第二轮专家评审修订；P0--P4 获准，P5--P6 需按缩减方案执行，P7 禁止实施

## 1. 研究主线与当前结论

下一阶段将研究问题收敛为：在统一射频约束、固定编辑预算和固定黑盒查询预算下，对真实 WiFi-B 样本进行参考条件的稀疏 token 编辑，生成能够降低冻结 Victim A 判别置信度、同时保持 Evaluator B 设备身份与物理可实现性的困难样本，用于后续防御训练与评估。

当前结论按以下口径冻结：

- VQ-VAE P1/K1024 只被认定为高保真重建与离散化 codec；尚未证明其 token 空间天然可编辑，也未达到 PPO-ready。
- 当前 causal RF-GPT 的长序列自由生成路线为 NO-GO；只保留为局部候选基线和 teacher-forced 诊断，不再扩展 2048-token 自回归生成。
- 稀疏 token 编辑为 PARTIAL PASS；已有结果证明“存在可攻击性”，但没有证明 RF-GPT 候选优于合理随机或近邻候选。
- 主线改为 `reference-conditioned sparse token editing / masked infill`。
- 威胁模型固定为 score-based black-box：搜索器可读取 Victim A 的 logits、置信度或 margin；不得读取梯度、参数或中间特征。Evaluator B 只做离线身份审计，不参与候选过滤、奖励或策略状态。

## 2. 数据隔离设计

历史事实是：现有 tokenizer、RF-GPT pilot 和 token-edit probe 已经根据完整 `reward_validation` 做过方法判断，因此对它重新切分只能产生开发集，不能产生独立 Gate。

当前数据清单包含三段此前未用于训练、模型选择或攻击评测的 `buffer`。实际缓存审计结果为：总计 1,779 条，每设备 94--111 条，每设备均有三个独立的时间间隔段。因此正式数据角色锁定为：

| 集合 | 来源与规模 | 唯一用途 |
|---|---:|---|
| tokenizer_dev | reward_validation 的开发子集 | 冻结 codec 复核与 token 图诊断；不声称独立 |
| generator_search_dev | reward_validation 的其余开发样本 | infill 选型、阈值锁定和 screening 调试 |
| policy_gate | untouched buffer，目标 50/device=850 | P5b/P6 的固定、一次性配对 Gate |
| defense_train_core | 原 defense_train=8,913 | 后续 defender training；不因 Gate 减少 |
| final_test | 原 final_test=8,317 | 全部方法冻结后只运行一次 |

拆分与选择规则：

1. `reward_validation` 内只建立开发角色；每设备按 `SHA256(seed | sample_id)` 稳定划分，避免运行库版本改变顺序。
2. `policy_gate` 从每设备的三个 buffer 时间段分层选择，目标配额为 16/17/17；段内仍按独立固定哈希排序，以覆盖不同时间位置并避免按 margin 挑样本。
3. Gate 入池样本必须在原始未编辑信号上同时被 Victim A 和 Evaluator B 正确分类。资格审计后才确定每段最终 ID。
4. 若任一设备不足 50 个合格样本，不做有放回采样，也不回退到 `reward_validation`。正式比较统一降为所有设备共同可满足的最小数量；只有 buffer 整体不足时才另行决定是否永久划出 `defense_train`。
5. 所有 sample ID、设备标签、原始 split、buffer 段、源索引、A/B clean 资格和哈希都写入 JSON/CSV；测试验证互斥、来源、类别计数和确定性。
6. P2/P3/P5a 的选型不得使用 `policy_gate` 的攻击结果。P5b/P6 所有方法共享相同 source IDs、种子、编辑数、查询预算、投影和约束。
7. P5/P6 指标明确命名为 `conditional valid-hard rate on dual-clean-correct sources`，不得外推为全分布攻击率。
8. `final_test` 继续封存；后续 defender training 使用完整 `defense_train_core`，不能只使用筛选后的 850 条 Gate 样本。

## 3. 阶段与关卡

### P0：冻结当前状态

目标：建立可追溯基线，避免后续结果口径漂移。

任务：

- 清点并计算 SHA256：数据 manifest、缓存元数据、冻结 Victim A、Evaluator B、VQ codec、RF-GPT pilot、现有 G0/G1/G4 报告、攻击基线和关键配置。
- 记录 Python、PyTorch、CUDA、GPU、操作系统、磁盘路径和测试结果。
- 不物理重命名已有 checkpoint；在 manifest 中建立逻辑角色和状态标记，例如 `frozen_codec`、`diagnostic_only`、`no_go_long_rollout`。
- 先检查仓库状态。若工作区不是 Git 仓库，则 `git_commit` 写为 `null`，以文件树哈希作为追踪依据；不擅自初始化 Git。

交付物：

- `reports/next_stage/current_state_manifest.json`
- `reports/next_stage/current_state_summary.md`

Gate P0：所有必需文件存在、哈希可复算、旧测试通过、manifest 不包含尚未冻结的实验结论。

### P1：开发角色与 untouched policy Gate

目标：建立无泄漏、可配对复现的评测基础。

任务：

- 为 `reward_validation` 建立 `tokenizer_dev` 和 `generator_search_dev` 稳定角色并保存 sample IDs；二者都标记为 historically seen development data。
- 识别每设备三个 buffer 时间段，为其分配稳定 `buffer_segment_id`。
- 对 buffer 只做一次双模型 clean-correct 资格审计；按段配额和稳定哈希选择 50/device，而不是按模型 margin 选择。
- 在 CSV 中保存 sample ID、device、buffer 段、原始索引、A/B clean prediction 与置信度、资格状态；不保存攻击结果。

交付物：

- `configs/data/wifib_next_stage_splits.json`
- `reports/next_stage/source_pool.csv`
- `tests/test_next_stage_splits.py`

Gate P1：开发角色与 buffer Gate 无交叠；policy Gate 只来自三个 untouched buffer；源池达到共同平衡数量；所有入池样本 A/B 均 clean-correct；`defense_train` 与 `final_test` 未访问。

### P2：Plausible token 候选图

目标：把“从 1,024 个码字中任意替换”收缩为可解释、版本化且所有方法共享的局部候选空间。

任务：

- 从冻结 codebook 建立 latent top-32 近邻图，同时保存 top-16 子集、距离和原 token 排除标记。
- 静态图只记录 latent distance、mutual-neighbor、code usage 和全局 transition/co-occurrence statistics；记录覆盖度、距离分位数、连通分量和互近邻比例。
- 不把单独码字的 decoded distance 当作物理合法性，因为 patch=1 仍不排除 decoder 上下文感受野。真实 source sequence 上的替换必须在运行时实际 decode，再由 P4 做局部/全帧 waveform precheck。
- P3 完成后，运行时候选与双向 infill 的相对概率门槛相交；P4 完成后再加入实际解码 RF precheck，避免 P2/P3 循环依赖。
- 每份图保存 codec hash、构建配置 hash、版本号和自身 hash。

交付物：

- `rffi_core/generators/token_candidates/build_codebook_neighbor_graph.py`
- `rffi_core/generators/token_candidates/token_candidate_graph.py`
- `artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz`
- `reports/next_stage/token_graph_diagnostics.json`
- `reports/next_stage/token_graph_diagnostics.md`
- `tests/test_token_candidate_graph.py`

Gate P2：图可确定性重建；候选索引无越界或自身环；活跃码字的覆盖、连通性和转移统计可报告；不存在 hash 漂移。P2 只通过“候选先验”关卡，不声称最终 waveform-valid。

### P3：参考条件局部 masked-infill

目标：在不查询 Victim/Evaluator 的训练阶段，学到真实上下文下的合理替换候选。

建议最小模型：256-token 双向 Transformer encoder，4 层、`d_model=128`、4 heads，输入包含 token、设备 ID、局部绝对位置、全帧归一化位置、mask 类型；输出 1,024 类 token logits。模型规模保持小型，先验证候选质量，不做长序列自由生成。

训练与验证：

- 训练只使用 `generator_train`；选型和早停只使用 `generator_search_dev`。
- mask 模式覆盖单 token、相邻 2 token 和短 span，并使全帧编辑率覆盖 0.1%、0.2%、0.5%、1%、2%、5%。
- 候选必须排除原 token，并同时满足 P2 图与相对概率门槛：`log p(q_j|context,y) >= log p(q_i|context,y) - delta_ll`，或等价概率比 `p(q_j)/p(q_i) >= rho`。`delta_ll/rho` 只在 `generator_search_dev` 锁定。
- P4 接入后，完整合法候选定义为 `latent/co-occurrence prior ∩ infill relative-probability ∩ actual decoded RF precheck`。交集为空时返回 no-op，不得硬凑候选数或退化到全码本随机。
- 若替代概率长期过度集中，优先加入 neighbor-corruption denoising：用合理近邻污染输入再恢复原 token；不因候选不足直接扩大模型。
- smoke test 先检查 loss、top-k 命中率、候选覆盖率、位置覆盖、设备条件使用和重复率，不调用 Victim A。

交付物：

- `rffi_core/generators/infill/models.py`
- `rffi_core/generators/infill/train_masked_infill.py`
- `rffi_core/generators/infill/evaluate_infill.py`
- `configs/generator/wifib_masked_infill_v1.yaml`
- `reports/next_stage/infill_validation.json`
- `reports/next_stage/infill_validation.md`

### P4：统一 RF 有效性与投影流水线

目标：所有替换器和搜索器使用同一数字域 waveform-validity 约束实现，且同时记录投影前后指标。当前阶段统一使用 `RF-constrained`、`digitally plausible` 或 `waveform-valid`，不宣称 physically realizable 或 over-the-air feasible。

指标：SNR、相对扰动功率、总信号 PAPR（主）、扰动 PAPR（辅）、归一化峰值增量、clipping、相对 clean waveform 的 EVM-like、PSD distance、22 MHz mask violation、Observable OOB Energy Ratio、局部不连续度/一阶差分。前导相关性只在窗口确实与前导结构对齐时计算；ACLR 只在采样频谱完整覆盖相邻信道时启用，不作为当前 35 MHz 数据的主指标。

统一流程：`token edit -> decode -> RF projection -> metric audit -> classifier query`。成功率只按投影后样本计算，投影前结果仅用于诊断。

交付物：

- `rffi_core/metrics/rf_validity_extended.py`
- `rffi_core/attacks/common_projection_pipeline.py`
- `tests/test_rf_validity_extended.py`
- `reports/next_stage/rf_metric_definition.md`

P3/P4 联合 Infill Gate 预注册四类门槛：

- 身份：Evaluator B pooled、macro、每设备最差值，以及相对 codec-only reconstruction 的额外 identity drop；
- RF：以 clean->codec、标准 RF augmentation 和带限 PGD 的分布为标尺，报告投影后违规率及连续指标分位数；
- 候选：经过三重过滤后仍存在候选的位置覆盖率和每位置有效候选数；不要求为满足数量而塞入低概率 token；
- 多样性：unique edit-mask ratio、replacement-token entropy、position entropy、每设备候选覆盖和单一 token/position domination。

具体数值阈值在基准分布测量后、第一次正式 Gate 前写入配置并锁定。Identity 初始判据为：B pooled 与 macro 均不低于 98%，无设备低于 95%，且相对 codec-only 的额外下降不超过 1--2 pp。RF pass line 由三类基准分布定义而不是凭空指定。若门槛不通过，回到候选图或 infill，不进入查询搜索。

### P5：两阶段位置选择 × 替换器实验

目标：先拆清收益来源，避免把位置选择、候选质量和查询优化混在一起。

P5a screening 只使用开发数据，不读取 policy Gate 的攻击结果：

- source：`generator_search_dev` 中双模型 clean-correct 的 10/device=170；
- seed：1；
- edit count：2、8、16，约等于 0.1%、0.4%、0.8%；
- query budget：64、256；
- methods：random+uniform plausible、random+nearest、random+infill、greedy+infill。

P5a 只回答 infill 是否有基本 proposal 价值、Greedy score 是否产生增益、哪几个 edit count 的 identity/RF 折中合理。根据预注册排序规则保留前 3--4 个方法，不做反复人工挑选。

P5b formal 才使用固定 untouched policy Gate：

- source：50/device=850，或 P1 确认的共同平衡数量；
- seeds：3；
- edit count：2、4、8、16；
- query budget：64、128、256；
- methods：仅 P5a 保留的前 3--4 个。

512 queries 只在 256 仍未出现性能饱和时补充；1,024 仅作扩展实验。41/102 token（约 2%/5%）不进入主表，只作为 stress test。

核心报告：投影后 conditional valid-hard rate、Victim A margin reduction、Evaluator B retention、waveform-valid pass、成功率-查询曲线、成功率-编辑数曲线、每设备 macro、有效样本数和 paired bootstrap 95% CI。

### P6：Greedy 与 Beam-Coordinate 黑盒搜索

目标：在不使用梯度和 B 的条件下，证明查询优化是否真正优于合理随机。

- 所有搜索先用不查询 Victim 的位置分数预排序：infill entropy、token surprisal、candidate spread 和 decoder local sensitivity。预排序规则与组合权重只在开发集锁定。
- 在 64-query 档，首轮默认评估 top-16 positions × top-4 replacements；后续动作后重新计算受影响局部位置。每档必须显式给出 `M × K × rounds <= Q` 的查询账本。
- Greedy：在预算内按 Victim margin 改善逐步选择，支持 early-stop 和 no-op。
- 第二搜索器锁定为 beam-coordinate search；它更贴合离散分层动作，并比同时实现 CEM 与 contextual bandit 更易控制查询数。CEM/Bandit 延后，不进入首轮主实验。
- 所有状态只包含原样本信息、历史动作、Victim A 输出、预算和 RF 合法性。
- Victim 查询在投影之后发生；缓存完全相同的候选查询，避免重复计费，同时记录原始请求数与去重后模型调用数。

Gate Q：满足以下任一条才允许进入 PPO：

1. 相同编辑与查询预算下，最佳查询搜索比 random plausible 的 valid-hard rate 至少高 5 个百分点，且配对 95% CI 不跨 0；或
2. 达到相同 valid-hard rate 时，所需查询数不超过 random plausible 的 50%。

若未通过，结论是 action/reward/RF/search 仍未建立 PPO 必要性；停止 PPO，回到 P2-P6 做诊断。

### P7：禁止实施，等待新的 PPO 规范

当前不得实现或训练 PPO。即使 Gate Q PASS，也必须先形成并评审新的 PPO spec：position head 的 reference prior 来自 infill entropy/candidate uncertainty；replacement head 初始化为 plausible candidate 上归一化的 infill logits，并只对此 head 定义明确 KL；stop head 单独建模并受 edit/query cost 约束。Evaluator B 继续完全排除在状态、奖励和在线过滤之外。只有新规范获批后，P7 状态才能从红灯改为条件执行。

### P8：条件性快速防御效用

仅在生成器与搜索器完全冻结后执行。比较 clean、RF augmentation、PGD-AT、random plausible、最佳非 RL 搜索，以及未来通过 Gate 和单独审批的 PPO。训练样本从完整 `defense_train_core` 生成，不能只使用 dual-clean-correct Gate 池。主要终点是未参与训练的隐藏攻击与跨模型鲁棒性，并报告全分布 clean/robust accuracy、clean accuracy 损失和每设备 macro。

Evaluator B 在 P0--P6 只作为 preliminary identity auditor，不能承担最终身份真值。P8/final 前需增加不同架构、不同输入表示且未参与 tokenizer/生成器选型的 Evaluator C，或进行外部域复验。ManyTx 可作为后续第二数据集验证，但当前不混入训练。

### 最终测试

只有当数据、模型、阈值、预算、统计脚本和报告模板均冻结后，才读取 `final_test` 一次。不得基于 final_test 结果回调模型、阈值或超参数。

## 4. 立即可执行的第一批工作

本批只执行 P0--P4，不包含正式 policy Gate 攻击、P5b/P6、PPO、Victim 驱动训练、final_test、ManyTx、tokenizer 重训或长序列 RF-GPT 扩展。

执行顺序：

1. P0 状态清点、哈希与摘要。
2. P1 开发角色、buffer segment 识别、单元测试、双模型资格审计和固定 policy Gate IDs；不在 Gate 上跑攻击。
3. P2 静态候选先验图与图诊断。
4. P3 masked-infill 最小模型、相对概率门槛、短 smoke run 和无 Victim 候选质量检查。
5. P4 waveform-validity 指标与统一投影/实际解码 precheck。
6. 在 `generator_search_dev` 上完成无 Victim infill 有效性评测；Evaluator B 只做一次性 preliminary identity audit。
7. 生成阶段 JSON/Markdown 报告并判定 Infill Gate。

资源安排：P0-P2 与 P4 以本地 CPU 为主；P3 先做本地小批量 smoke。只有 smoke、数据读取和指标流水线全部通过后，才考虑把项目的必要子集同步到 AutoDL `/tmp/exp7`。远程环境必须先复核 Python、CUDA 与 GPU；若仍无可用 GPU，则不上传数据和 checkpoint。

## 5. 统一报告协议

每个阶段同时输出 JSON 和 Markdown，至少包含：

- `stage`
- `git_commit` 或文件树哈希
- `config_hash`
- `data_split_hash`
- `checkpoint_hash`
- `seeds`
- `source_count`
- `victim_query_budget`
- `edit_budget`
- pooled、per-device macro、每设备计数和 paired bootstrap 95% CI
- `gate.status` 与 `gate.reason`
- `artifacts`
- `next_allowed_stage`
- `prohibited_actions`

## 6. 主要风险与应对

| 风险 | 早期信号 | 停止/修正策略 |
|---|---|---|
| buffer 中双模型 clean-correct 样本不足 | 某设备少于 50 | 不补样、不回退历史 validation；统一降低每设备数量，整体不足才讨论 defense_train |
| token 近邻在时域仍产生尖峰 | 局部差分/PAPR 尾部异常 | 加严 decoded-distance 与 continuity filter，不扩大码本随机范围 |
| infill 只复制原 token | top-k 被原 token 垄断、候选为空 | 训练时增加替代目标/对比约束；评测时仍强制排除原 token |
| infill 候选自然但没有攻击性 | RF/B 很好但 Victim margin 几乎不降 | 保留生成器，转由 P6 的位置/候选查询搜索验证；不直接上 PPO |
| Greedy/beam-coordinate 不优于随机 | Gate Q1/Q2 均失败 | 停止 PPO，诊断候选覆盖、奖励尺度、预算和投影损失 |
| 投影抹除 token 编辑效果 | pre/post margin 差距大 | 将投影置于所有查询之前，调整候选图和 RF 阈值，而不是报告投影前成功 |
| policy Gate 被反复调参污染 | 多次依据 Gate 改超参数 | `generator_search_dev` 完成 screening/调参；policy Gate 只做预注册正式配对运行 |
| 远程环境不可用 | 无 GPU/Python/CUDA | 继续本地 smoke；不上传大数据，不影响远程其他实验 |

## 7. 已锁定决定与剩余批准边界

1. 数据：`reward_validation` 只作开发；policy Gate 优先且当前可从 1,779 条 untouched buffer 中建立；`defense_train` 保持完整；`final_test` 封存。
2. 规模：P5a 使用 10/device、1 seed、edit 2/8/16、Q=64/256；P5b 使用 50/device、3 seeds、edit 2/4/8/16、Q=64/128/256，只保留 3--4 个方法。512/1,024 与 2%/5% 均不进入主表。
3. 候选：`latent/co-occurrence prior ∩ infill relative-probability ∩ actual decoded RF precheck`；无候选即 no-op。
4. 搜索：首版只实现 Greedy 与 beam-coordinate，并为每个预算显式定义无 Victim 的 position pre-ranking 与查询账本。
5. P7：当前禁止实施；Gate Q PASS 只是提交 PPO 三-head reference spec 的必要条件，不是自动训练授权。

P0--P4 可以按本计划开始；P5a 需在 P0--P4 Gate PASS 后开始；P5b/P6 必须使用预注册协议；P7 需要新的专家批准。
