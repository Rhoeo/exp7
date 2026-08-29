# RFFI 防御型困难样本生成：下一阶段 Codex 执行规范

> 日期：2026-08-28  
> 工作区：`E:\exp7`  
> 基础报告：`EXPERIMENT_PROGRESS_REPORT_2026-08-28.md`  
> 文档目的：明确当前实验结论、主线调整、允许实施的任务、暂时禁止的任务、Go/No-Go Gate 和交付物。  
> **发生冲突时，本文件优先于此前关于 full RF-GPT、无条件生成和直接 PPO 的旧计划。**

---

## 0. 研究定位与边界

本项目研究的是：

> **面向 RFFI 防御训练的受约束困难样本生成。**

生成样本只用于：

- 暴露冻结 RFFI 模型的脆弱区域；
- 建立 hard-sample pool；
- 训练/微调 defender；
- 评价 unseen、hidden 和 cross-model 鲁棒性。

最终目标不是单纯最大化攻击成功率，而是：

\[
\max \Delta \mathrm{RobustAcc}_{hidden}
\]

同时满足：

\[
\mathrm{RFViolation}\le \tau_{RF}
\]

\[
\mathrm{IdentityDrift}\le \tau_{ID}
\]

\[
\Delta \mathrm{CleanAcc}\ge-\tau_C
\]

---

# 1. 当前事实：已经完成了什么

以下结果已经存在，禁止重复从头实现，除非发现可复现性错误。

## 1.1 数据与缓存

当前配置：

```text
configs/data/rffi_data_v1.json
```

WiFi-B：

```text
17 devices
59,409 frames
raw complex-IQ length = 17,550
window start = 128
window length = 2,048
```

缓存：

```text
WiFi-B IQ:
E:\data_cache\rffi_v1\...
shape = (59,409, 2, 2,048)

WiFi-B RF tokens:
E:\data_cache\rffi_v1\tokens\wifib_vq_p1_k1024\tokens.npy
shape = (59,409, 2,048)
dtype = uint16
```

现有 split：

```text
generator_train   = 31,486
reward_validation = 8,914
defense_train     = 8,913
final_test        = 8,317
```

ManyTx 已缓存，但**尚未进入当前生成器主线**。

---

## 1.2 Stage G0：冻结分类器

路径：

```text
runs/stage_g0/frozen/wifib_v1/
```

模型：

```text
Victim A:
  time-domain residual CNN
  reward-validation accuracy = 0.9054
  final-test accuracy        = 0.8670

Evaluator B:
  FFT multi-scale CNN
  reward-validation accuracy = 0.9973
  final-test accuracy        = 0.9859
```

当前角色必须重新冻结为：

```text
Victim A:
  Generator 可以查询 score/logits
  不允许读取梯度
  不允许更新参数
  不建议读取 intermediate embedding

Evaluator B:
  仅用于 validation identity check
  不参与 proposal
  不参与 reward
  不参与 position/replacement selection
```

---

## 1.3 Stage G1：VQ codec

冻结检查点：

```text
runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt
```

SHA-256：

```text
331ff5136efd8ed45ef444b348e7bb51b0b77a8e1da5523a4fbeb40ae0e3607e
```

完整 `reward_validation` 结果：

```text
reconstruction SNR              = 30.36 dB
waveform correlation            = 0.9995
spectral log-L1                 = 0.01267
Victim A accuracy drop          = 1.63 pp
Victim A identity retention     = 0.9689
Evaluator B accuracy drop       = 0.00 pp
Evaluator B identity retention  = 0.9997
active codes                    = 1024 / 1024
token perplexity                = 981.19
sequence length                 = 2,048
```

### 当前结论必须写成

```text
PASS:
  high-fidelity reconstruction codec

NOT YET PROVEN:
  generative semantic tokenizer
  editable tokenizer
  PPO-ready tokenizer
```

不要再笼统写：

```text
“VQ tokenizer 已完全可行”
```

更准确的表述是：

> `patch=1, K=1024` 已通过重构与初步身份保持 Gate；但它更接近高精度 sample-level codec，尚未证明其 token 空间适合自由生成或物理合理编辑。

---

## 1.4 Stage G3：当前 causal RF-GPT

模型：

```text
d_model  = 96
layers   = 3
heads    = 4
context  = 256
params   ≈ 575k
```

结果：

```text
teacher-forced perplexity = 10.83
token accuracy            = 28.3%
correct-condition NLL     = 2.39418
shuffled-condition NLL    = 2.65873
condition advantage       = 0.26455 nat
```

说明模型学习了：

- 局部 token transition；
- device condition；
- teacher-forced next-token distribution。

但 2,048-token 自由 rollout 失败：

```text
sampling:
  Victim A identity ≈ 17.6%
  Evaluator B identity = 0%

greedy:
  A/B ≈ 11.8%

256-token prefix continuation:
  Victim A ≈ 11.8%
  Evaluator B ≈ 5.9%
```

### 当前正式结论

\[
\boxed{\text{Unconditional / long-rollout RF-GPT = NO-GO}}
\]

当前 causal RF-GPT 可以保留为：

```text
local proposal baseline
teacher-forced diagnostic
```

但不能作为当前 PPO reference policy 直接继续 full training。

---

## 1.5 Stage G4：参考 token 编辑

现有 pilot：

```text
68 samples
4 samples/device
only clean-correct sources
22 MHz centered band projection
```

结论：

- RF-GPT 稀疏编辑可以产生部分 valid hard samples；
- 但在当前 pilot 中未超过均匀随机 token 替换；
- 任意 token 替换产生高扰动 PAPR；
- PGD 是 white-box upper bound，不是同 threat-model 的唯一公平对照。

### 当前正式结论

```text
Token edit action space:
  PARTIAL PASS

RF-GPT proposal superiority:
  NOT PROVEN

PPO readiness:
  NOT REACHED
```

---

# 2. 现在冻结的研究主线

下一阶段不再追求：

```text
device ID
  ↓
从零自由生成 2,048 个 RF tokens
  ↓
完整 IQ
```

主线调整为：

\[
\boxed{\text{Reference-Conditioned Sparse Token Editing / Infill}}
\]

结构：

```text
Original IQ
    ↓
Frozen VQ Codec
    ↓
Reference token sequence q
    ↓
Select a small position/span set
    ↓
Masked RF Infill Model proposes plausible alternatives
    ↓
Decode edited tokens
    ↓
Common RF projection + validity checks
    ↓
Query Victim A score
    ↓
Margin feedback
```

最终 PPO 版本仅在非 RL query search 证明动作空间有效后启动：

```text
Frozen Reference Infill Model
             ↓ copy
┌────────────────────────────┐
│                            │
Reference Policy        PPO Token Editor
│                            │
└──────── policy KL ──────────┘
                             ↓
          position/span + replacement + stop
                             ↓
                        edited IQ
```

---

# 3. 当前威胁模型

主线统一为：

\[
\boxed{\text{Score-Based Black-Box Hard-Sample Generation}}
\]

Generator 允许：

```text
query Victim A
read logits / confidence / margin
```

Generator 禁止：

```text
read Victim A gradients
backpropagate through Victim A
update Victim A
read Evaluator B feedback
use final_test
```

PGD 的角色固定为：

```text
white-box upper bound
+
PGD adversarial-training baseline
```

不能仅用：

```text
PGD 100% vs token editor x%
```

直接否定黑盒生成路线。

---

# 4. 当前必须做什么

---

## Phase P0：冻结当前状态和结果

### MUST

1. 保留并校验现有 G0/G1 checkpoints；
2. 保存当前 Git commit、config、split hash、checkpoint hash；
3. 将当前状态写入 machine-readable manifest；
4. 不覆盖已有 report；
5. 将现有模型重命名/标记为：

```text
vq_p1_k1024_reconstruction_codec
rfgpt_causal_pilot_local_proposal
```

### Deliverable

```text
reports/next_stage/current_state_manifest.json
reports/next_stage/current_state_summary.md
```

---

## Phase P1：重构 validation protocol

当前 `reward_validation` 已被 tokenizer、GPT pilot 和 token-edit probe 多次使用。

必须进一步拆分，避免继续对同一 validation set 过拟合。

### 建议逻辑拆分

```text
tokenizer_validation
generator_validation
policy_validation
```

三者均从当前 `reward_validation` 派生，使用固定 seed 和 stratified-by-device split。

建议比例：

```text
tokenizer_validation ≈ 3,000
generator_validation ≈ 3,000
policy_validation    ≈ 2,914
```

具体数量允许根据设备均衡做小幅调整。

### MUST

- 输出每条样本的固定 ID；
- 保证设备分层；
- 保证 split 不重叠；
- 之后不再改变；
- `final_test` 继续封存。

### 固定 source pool

对 edit/search benchmark：

```text
50~100 clean-correct samples per device
```

建议首轮：

```text
50/device
17 devices
≈ 850 source samples
```

Source sample 必须同时满足：

```text
Victim A clean prediction = true label
Evaluator B clean prediction = true label
```

所有方法使用完全相同的 source IDs。

### Deliverables

```text
configs/data/wifib_next_stage_splits.json
reports/next_stage/source_pool.csv
tests/test_next_stage_splits.py
```

---

## Phase P2：建立 Plausible Token Candidate Graph

当前随机 baseline 从整个 1,024 codebook 任意替换，会造成：

- 大 token jump；
- 采样点级尖峰；
- 不自然 transition；
- 极高 perturbation PAPR。

下一步必须把 action space 限制为“合理邻居”。

### 每个 token 的候选应综合

1. **Codebook latent distance**

\[
d_z(q_i,q_j)=\|e_i-e_j\|_2
\]

2. **Decoded local waveform distance**

替换单个 token 后，比较局部解码窗口：

\[
d_x(q_i,q_j|\text{context})
\]

3. **Transition compatibility**

结合左右参考 token：

\[
\log p(q_j|q_{i-k:i-1},q_{i+1:i+k},y)
\]

### V1 实现建议

离线构建：

```text
latent top-k neighbor graph
```

例如：

```text
k_latent = 16 or 32
```

运行时再用：

```text
local waveform distance
+
infill transition probability
+
RF validity precheck
```

过滤候选。

### MUST

- 原 token 不计入 replacement；
- Random baseline 也必须从同一 candidate set 采样；
- GPT/infill、Greedy、CEM、PPO 使用同一候选空间；
- 保存 candidate graph version 和 hash。

### Deliverables

```text
rffi_core/generators/rfgpt/build_codebook_neighbor_graph.py
rffi_core/generators/rfgpt/token_candidate_graph.py
artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz
reports/next_stage/token_neighbor_diagnostics.md
tests/test_token_candidate_graph.py
```

---

## Phase P3：训练 Reference-Conditioned Masked RF Infill Model

不要继续扩大 2,048-token causal rollout。

实现一个只恢复少量 masked token/span 的双向模型。

### V1 输入

```text
256-token local window
device ID
absolute/normalized position
masked position(s)
left and right reference context
```

### 为什么使用 256-token local window

- 复用当前 context 规模；
- 避免 full 2,048 attention；
- 推理时参考序列的左右上下文都存在；
- 不需要 2,048 步 rollout；
- 设备身份由原始序列锚定。

### 训练 mask ratio

至少覆盖：

```text
0.1%
0.2%
0.5%
1%
2%
5%
```

由于 local window 为 256，实际实现可映射为：

```text
1 token
2 tokens
short spans
```

并在 full-frame 上通过随机位置采样形成不同全局 edit ratio。

### 模型输出

对 masked position 输出：

```text
token logits
top-k alternatives
```

必须排除原 token后，再与 plausible candidate graph 取交集。

### 训练阶段不得使用

```text
Victim A reward
Evaluator B reward
final_test
```

### Infill Gate

在不查询 Victim A 的情况下，编辑必须满足：

- 高 Evaluator B identity retention；
- 低 RF violation；
- 相比 full-codebook random 显著降低峰值/尖峰；
- token alternatives 有多样性；
- 不发生大规模 duplicate/collapse。

### Deliverables

```text
rffi_core/generators/rfgpt/masked_infill_model.py
rffi_core/generators/rfgpt/train_masked_infill.py
rffi_core/generators/rfgpt/evaluate_masked_infill.py
configs/generator/wifib_masked_infill_v1.yaml
reports/next_stage/masked_infill_report.md
```

---

## Phase P4：补齐 RF Validity Metrics

22 MHz band projection 保留为所有方法的共同 postprocess，但它不是完整的 RF 合法性证明。

### 所有方法统一记录

1. Perturbation SNR；
2. Relative perturbation power；
3. Perturbation PAPR；
4. **Final total-signal PAPR**；
5. Normalized peak perturbation：

\[
\frac{\max|\delta|}{RMS(x)}
\]

6. clipping ratio；
7. EVM-like metric；
8. PSD distance；
9. 22 MHz mask violation；
10. ACLR（若可稳定实现）；
11. preamble correlation；
12. local discontinuity / derivative metric。

### MUST

- 同时保存 projection 前后指标；
- success 只按 projection 后样本判断；
- 不仅报告 perturbation PAPR；
- 所有 baseline 走相同 projection 和 validity pipeline。

### Deliverables

```text
rffi_core/metrics/rf_validity_extended.py
rffi_core/attacks/common_projection_pipeline.py
tests/test_rf_validity_extended.py
reports/next_stage/rf_metric_definition.md
```

---

## Phase P5：做公平的 Factorial Edit Benchmark

当前“GPT 改 41~102 token、Random 改 2~6 token，仅匹配 SNR”的比较不能作为最终方法优劣结论。

下一轮必须拆开：

### 位置选择

```text
Random Position
Victim-Score Greedy Position
CEM/Bandit Position
PPO Position       # 仅未来
```

### Replacement Proposal

```text
Uniform from same plausible-neighbor set
Nearest Codebook Neighbor
Masked-Infill top-k
PPO Replacement    # 仅未来
```

形成至少以下组合：

| Position selector | Replacement proposer |
|---|---|
| Random | Uniform plausible neighbor |
| Random | Nearest neighbor |
| Random | Infill top-k |
| Greedy | Uniform plausible neighbor |
| Greedy | Infill top-k |
| CEM/Bandit | Infill top-k |

### 公平约束

所有组合必须固定：

```text
same source IDs
same edit count
same candidate graph
same Victim query budget
same RF projection
same RF constraints
same random seeds
```

额外报告：

```text
valid-hard rate vs edit count
valid-hard rate vs Victim query count
margin reduction vs query count
RF validity vs query count
```

### Deliverables

```text
rffi_core/search/token_edit_benchmark.py
configs/search/edit_factorial_v1.yaml
reports/next_stage/edit_factorial_results.csv
reports/next_stage/edit_factorial_report.md
```

---

## Phase P6：先实现非 RL Query-Based Search

PPO 之前，先证明 victim score 对当前 token action space 有可利用信号。

至少实现：

```text
Greedy Coordinate Search
CEM or Contextual Bandit Search
```

### 推荐 Greedy V1

每步：

1. 从未编辑 position 中采样/筛选少量候选；
2. 对每个 position 获取 plausible replacements；
3. 查询 Victim A margin；
4. 选择 margin reduction 最大且 RF constraints 合法的 action；
5. 达到 success 或 edit/query budget 后停止。

### 推荐 CEM/Bandit V1

动作可拆成：

```text
position
replacement candidate index
stop
```

但不允许直接在：

\[
2048\times1024
\]

全空间搜索。

### PPO 启动 Gate

只有满足以下至少一项，才允许进入 PPO：

#### Gate Q1：效果优势

在相同 source pool、相同 edit count、相同 RF constraints、相同 query budget 下：

```text
best query-based search
-
random plausible-neighbor search
>= 5 percentage points valid-hard rate
```

#### Gate Q2：查询效率优势

达到相同 valid-hard rate 时：

```text
query-based search queries
<= 50% of random search queries
```

5 pp 和 50% 是当前工程止损阈值，不是行业统一标准；必须在 report 中标注为 project gate。

### Gate 未通过时

禁止启动 PPO。

优先诊断：

```text
candidate graph
position sensitivity
replacement quality
reward signal
RF constraints
```

### Deliverables

```text
rffi_core/search/greedy_token_editor.py
rffi_core/search/cem_token_editor.py
configs/search/query_search_v1.yaml
reports/next_stage/query_search_report.md
reports/next_stage/query_search_gate.json
```

---

## Phase P7：条件满足后才实现 PPO Token Editor

只有 `query_search_gate.json` 明确 PASS 后才开始。

### PPO reference policy

使用冻结的 masked infill model：

\[
\pi_{\mathrm{ref}}
\]

PPO policy：

\[
\pi_\phi \leftarrow \pi_{\mathrm{ref}}
\]

### Action 设计

采用分层动作，而不是完整 codebook：

```text
Action 1: choose position or short span
Action 2: choose replacement from top-k plausible alternatives
Action 3: stop
```

### State

仅使用 black-box 可用信息：

```text
reference/edited token summary
current edit mask
current operator/action history
Victim A output score/margin
RF validity summary
remaining edit budget
remaining query budget
step index
```

默认不使用 Victim A intermediate embedding。

### Reward

\[
r_t=
\lambda_h\Delta H_t
-\lambda_{KL}D_{KL}(\pi_\phi||\pi_{ref})
-\lambda_{peak}V_{peak}
-\lambda_{spec}V_{spec}
-\lambda_{dist}D(x,x')
-\lambda_{edit}N_{edit}
\]

其中：

\[
H_t=\max_{j\ne y}z_j-z_y
\]

终止 bonus 只在同时满足以下条件时给予：

```text
Victim A becomes hard/misclassified
Evaluator-independent RF constraints pass
edit/query budget respected
```

注意：训练 reward 中不能查询 Evaluator B。

Evaluator B 只能在 episode 结束后做 validation 统计，不回传给 PPO。

### PPO 必须比较

```text
Random plausible-neighbor
Greedy
CEM/Bandit
PPO
```

同一：

```text
query budget
edit budget
source pool
candidate graph
RF constraints
```

### PPO Gate

PPO 必须至少满足一项：

```text
higher valid-hard rate than best non-RL search
OR
same valid-hard rate with fewer queries
OR
same generation quality with much lower per-sample online search cost after amortization
```

否则 PPO 不作为主方法。

---

## Phase P8：Quick Defense Utility

只有生成器/搜索方法冻结后才开始。

生成相同数量样本，统一 defender training budget。

比较：

```text
Clean Training
Standard RF Augmentation
PGD-AT
Random Plausible-Neighbor Edit
Best Non-RL Query Search
PPO Editor              # 仅 Gate 通过后
```

统一：

```text
clean/hard ratio
optimizer
epochs
batch size
seed set
source sample count
```

### 评价

```text
Clean Accuracy
Seen Robust Accuracy
Hidden Robust Accuracy
Cross-Model Robust Accuracy
RF Validity
```

### 核心判断

即使某方法 valid-hard rate 高，如果：

```text
hidden/cross-model defense gain ≈ 0
```

则不能称为有效防御样本生成方法。

---

# 5. 当前明确不做什么

以下事项当前全部禁止。

## 5.1 不训练 full unconditional RF-GPT

禁止：

```text
扩大当前 causal RF-GPT
增加大量 epoch
直接 context=2048 full training
继续从 device ID 自由生成完整 2,048 token
```

原因：

- teacher forcing 已有效；
- long rollout 身份失败；
- 当前主要问题不是模型规模不足的直接证据；
- 扩大模型可能只会更昂贵地重复 exposure-bias 问题。

---

## 5.2 不在当前 causal RF-GPT 上启动 PPO

禁止：

```text
PPO fine-tune current free-running RF-GPT
```

原因：

- reference policy 自由生成已离开真实 RF/identity manifold；
- 强 KL 会把 PPO 锁在坏分布；
- 弱 KL 会放大 reward hacking。

---

## 5.3 不使用 final_test 做任何选择

禁止用 `final_test`：

- 选择 tokenizer；
- 选择 infill model；
- 选择 candidate graph；
- 选择 search method；
- 调 reward；
- 调 PPO；
- early stopping；
- LLM feedback。

只有整套方法、约束、超参数和 seed protocol 全部冻结后，才能一次性运行 final test。

---

## 5.4 不把 Evaluator B 放入生成闭环

禁止：

```text
use B logits for proposal
use B gradient
use B score in reward
use B for action filtering during training
```

B 只做 validation identity check。

最终论文还需要独立 Evaluator C 或外部域验证；当前不要把 B 称为绝对 identity oracle。

---

## 5.5 不从整个 1,024 codebook 做任意替换

Random、Infill、Greedy、CEM、PPO 必须共享 plausible candidate graph。

禁止：

```text
uniform random replacement over all 1024 codes
```

作为最终公平 baseline。

现有 full-codebook random 结果保留为 diagnostic baseline。

---

## 5.6 不继续使用 68 条样本下百分比做路线结论

68 条 pilot 只能用于 smoke test。

后续主结论至少使用：

```text
≈ 850 fixed clean-correct source samples
```

并报告 confidence interval。

---

## 5.7 不只按 SNR 匹配方法

公平比较必须同时控制：

- source IDs；
- edit count；
- candidate space；
- query budget；
- RF constraints；
- seed。

SNR 只是一个指标，不是唯一公平条件。

---

## 5.8 不立即重做 tokenizer

当前 VQ codec 保留为主线 reconstruction codec。

暂时不立即投入：

```text
FSQ
RVQ
product VQ
new compressed tokenizer
```

只有在以下情况才开启 tokenizer redesign：

```text
plausible-neighbor + infill 仍频繁产生非法尖峰
OR
可编辑性 Gate 持续失败
OR
2,048-token representation 成为明确计算瓶颈
```

届时再做：

```text
current VQ vs FSQ vs RVQ
```

并重新跑完整 identity Gate。

---

## 5.9 不立即混入 ManyTx

ManyTx 当前不进入 WiFi-B generator training。

先在 WiFi-B 上跑通：

```text
reference edit
→ query search
→ optional PPO
→ quick defense
```

之后 ManyTx 用于：

- 第二数据集复验；
- 跨域泛化；
- 预训练可行性研究。

---

## 5.10 不因 AutoDL 可用就立即做大训练

当前方法 Gate 未完成前，不上传全量数据做 full training。

需要 GPU 的正确时点：

```text
masked infill smoke test passed
candidate graph fixed
evaluation protocol fixed
```

---

# 6. 评价定义

## 6.1 Source eligibility

一条 source sample 只有同时满足：

```text
Victim A clean correct
Evaluator B clean correct
```

才进入主 benchmark。

---

## 6.2 Hardness

连续指标：

\[
H(x)=\max_{j\ne y}z_j-z_y
\]

Margin improvement：

\[
\Delta H=H(x')-H(x)
\]

---

## 6.3 Valid hard sample

V1 定义：

```text
Victim A prediction after edit != y
AND
Evaluator B prediction after edit == y
AND
all RF constraints pass
AND
edit/query budget respected
```

同时额外报告 near-hard samples：

```text
margin reduction >= predefined threshold
but Victim A not yet flipped
```

---

## 6.4 统计方式

必须同时报告：

```text
pooled average
per-device macro average
per-device counts
paired bootstrap 95% CI
valid-hard rate vs query budget
valid-hard rate vs edit count
```

---

# 7. 建议代码目录增量

在现有工程上增加：

```text
rffi_core/
├── generators/
│   └── rfgpt/
│       ├── build_codebook_neighbor_graph.py
│       ├── token_candidate_graph.py
│       ├── masked_infill_model.py
│       ├── train_masked_infill.py
│       └── evaluate_masked_infill.py
│
├── search/
│   ├── token_edit_benchmark.py
│   ├── greedy_token_editor.py
│   ├── cem_token_editor.py
│   └── search_metrics.py
│
├── rl/
│   └── ppo_token_editor/
│       ├── env.py
│       ├── policy.py
│       ├── trainer.py
│       └── evaluator.py
│
├── metrics/
│   └── rf_validity_extended.py
│
└── defense/
    └── quick_defense_eval.py
```

---

# 8. 实施顺序与停止条件

严格按以下顺序：

```text
P0  Freeze current artifacts
 ↓
P1  Split protocol + fixed source pool
 ↓
P2  Plausible token candidate graph
 ↓
P3  Masked reference-conditioned infill
 ↓
P4  Extended RF validity
 ↓
P5  Fair factorial benchmark
 ↓
P6  Greedy + CEM/Bandit query search
 ↓
Gate Q
 ├── FAIL → diagnose action space; no PPO
 └── PASS
        ↓
P7  PPO token editor
        ↓
Gate PPO
 ├── FAIL → retain best non-RL search
 └── PASS
        ↓
P8  Quick defense utility
        ↓
Freeze all protocols
        ↓
Final test once
```

任何阶段失败时：

- 保存结果；
- 生成失败分析；
- 不自动跳过 Gate；
- 不通过修改 final-test 或放宽 RF constraints“制造成功”。

---

# 9. Codex 每阶段报告格式

每个阶段结束必须输出：

```yaml
stage:
git_commit:
config_hash:
data_split_hash:
checkpoint_hash:
random_seeds:
source_count:
victim_query_budget:
edit_budget:
metrics:
gate:
  status: PASS | FAIL | INCONCLUSIVE
  reason:
artifacts:
next_allowed_stage:
prohibited_actions:
```

并生成：

```text
machine-readable JSON
human-readable Markdown
```

---

# 10. 当前可以陈述与不能陈述的结论

## 可以陈述

1. WiFi-B 上已经实现高保真离散 VQ reconstruction；
2. 重构后 A/B identity 基本保持；
3. 当前 causal RF-GPT 学到了局部 token 语法和设备条件；
4. long-rollout 自由生成不能保持设备身份；
5. token edit 空间存在能降低 Victim A margin 的动作；
6. 当前 RF-GPT proposal 尚未证明优于随机；
7. PPO 尚未启动，也尚未达到启动 Gate。

## 不能陈述

1. “RF-GPT 已经能够可靠生成指定设备 IQ”；
2. “RF-GPT 编辑优于随机 baseline”；
3. “VQ token 天然物理合理”；
4. “Evaluator B 证明真实设备身份绝对保持”；
5. “PPO 会解决当前自由生成失败”；
6. “本方法已经优于 PGD”；
7. “当前 68 样本结果具有统计显著性”；
8. “最终 test 已证明泛化”。

---

# 11. 当前第一批 Codex 任务

Codex 现在只执行以下任务：

```text
1. 创建 current_state_manifest；
2. 拆分 reward_validation；
3. 建立固定 850 左右 source pool；
4. 实现 plausible token candidate graph；
5. 扩展统一 RF validity metrics；
6. 实现 256-token local masked infill smoke model；
7. 运行 no-victim infill validity test；
8. 生成阶段报告并等待 Gate 判断。
```

Codex 现在不要执行：

```text
full causal RF-GPT training
context=2048 scaling
PPO
final_test
ManyTx integration
tokenizer replacement
defender full training
```

---

# 12. 最终工程目标

本阶段成功时，系统应达到：

```text
真实 IQ
  ↓
高保真 VQ codec
  ↓
参考 token
  ↓
受约束 local infill candidates
  ↓
score-based query search
  ↓
合法、身份保持、可控的 hard samples
```

只有这一层稳定后，才将 search amortize 成 PPO policy：

```text
query search evidence
  ↓
PPO token editor
  ↓
hard-sample pool
  ↓
defender training
  ↓
hidden/cross-model robustness
```

本阶段最重要的原则：

\[
\boxed{
\text{先证明“参考条件下可合理编辑”，再证明“查询反馈有用”，最后才训练 PPO。}
}
\]
