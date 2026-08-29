# 保守版：RFFI 防御导向的 RF-Operator PPO + LLM Reward Evolution 技术方案

> 版本日期：2026-08-27  
> 用途：供 Codex 直接理解并实施。  
> 研究定位：**面向射频指纹识别（RFFI）防御训练的困难样本生成**，不以现实攻击实施为研究目标。  
> 优先级：**可复现性 > 可训练性 > 物理可解释性 > LLM 新颖性**。

## 1. 一句话定义

本方案不让 LLM 直接生成 I/Q，也不让 PPO 直接控制每一个 I/Q 采样点，而是：

\[
\boxed{\text{LLM 外层设计 Reward}+\text{PPO 内层搜索低维 RF 变换参数}+\text{Defense Utility 评价样本}}
\]

真正生成 I/Q 的部分是受约束的 RF transformation：

\[
x'=T(x;\theta),\qquad \theta=\pi_\phi(s)
\]

而不是让 PPO 直接输出 \(\delta\in\mathbb R^{2L}\)。

## 2. 与 AdvTG 的关系

该方案只保留 AdvTG 中容易迁移且实验风险较低的思想：

- victim feedback 驱动困难样本生成；
- PPO 学习可复用生成策略；
- 对合法性施加领域约束；
- 用生成结果反馈生成策略。

但它**不是 AdvTG 的同构迁移**。

AdvTG：

```text
Domain LLM
   ↓ copy
RL-tuned LLM
   ↑
  PPO
   ↑
detector reward
```

本方案：

```text
LLM
 ↓
Reward configuration
 ↓
PPO MLP policy
 ↓
RF operator parameters
 ↓
IQ hard samples
 ↓
victim / RF metrics
```

因此论文中应表述为“受 RL-guided domain-constrained generation 启发”，不能写成“直接将 AdvTG 迁移到 RFFI”。

## 3. 核心科学问题

> 在保持设备身份语义和 RF 有效性的条件下，能否学习一种可复用的困难样本搜索策略，并用这些样本提升 RFFI 模型对未见扰动的鲁棒性？

最终目标不是单纯最大化 fooling rate，而是：

\[
\max \Delta RA_{\text{holdout}}
\]

满足：

\[
V_{\text{RF}}\le\tau_v,\qquad \Delta Acc_{\text{clean}}\ge-\tau_c
\]

## 4. 总体流程

```text
Human Reward R0
      ↓
PPO-Fixed baseline
      ↓
Experiment Summary M0
      ↓
LLM
      ↓
Reward candidates
      ↓
short PPO
      ↓
hard sample pool
      ↓
RF validity + quick defender
      ↓
holdout defense gain
      ↓
next LLM round
```

初始化时：**人工 Reward → PPO → LLM**。  
正式 outer loop：**LLM → Reward → PPO → Defense Evaluation → LLM**。

## 5. RFFI 扰动空间必须分层

### Tier A：V1 优先允许

更偏 nuisance 的 operator：

- global phase rotation；
- bounded AWGN / colored-noise strength；
- small timing shift；
- mild multipath；
- band-limited additive component；
- global gain（仅在 preprocessing 不会消除它时）。

### Tier B：默认禁止进入 V1 training

可能本身就是设备指纹来源：

- CFO；
- IQ gain imbalance；
- IQ phase imbalance；
- PA nonlinearity；
- phase noise；
- sampling frequency offset。

只有 identity-preservation sweep 通过后才能启用。

### Tier C：逐点 additive IQ perturbation

用于 FGSM/PGD/UAP baseline，但**不作为 PPO V1 的高维 action**。

## 6. PPO Environment

Episode：

\[
x_0=x,\quad \theta_0=0
\]

每一步：

\[
a_t\sim\pi_\phi(a|s_t)
\]

\[
\theta_{t+1}=clip(\theta_t+\Delta\theta_t)
\]

\[
x_{t+1}=T(x;\theta_{t+1})
\]

建议 `episode_steps = 6`，后续做 4/6/8 消融。

## 7. State

\[
s_t=[e_t,c_t,q_t,\theta_t,t/T]
\]

其中：

- `e_t`：frozen victim embedding，经固定 PCA/random projection 压到 16~32 维；
- `c_t`：true-class confidence、top competitor、logit margin、CE loss；
- `q_t`：relative power、SNR-like、EVM-like、PSD distance、constraint margin；
- `theta_t`：当前 operator 参数。

logit margin：

\[
m_t=z_y-\max_{j\neq y}z_j
\]

## 8. V1 Action

建议先从 6 维开始：

```text
a0 = Δglobal_phase
a1 = Δnoise_power
a2 = Δtiming_shift
a3 = Δmultipath_amplitude
a4 = Δmultipath_phase
a5 = Δmultipath_delay
```

operator sweep 后再决定是否增加 gain / band-limited noise。

## 9. Reward

初始人工 reward：

\[
H_t=-m_t
\]

\[
r_t=
w_h(H_t-H_{t-1})
-w_p\tilde P_t
-w_e\tilde E_t
-w_s\tilde S_t
-w_a\|a_t\|^2
\]

若 episode 最终形成 valid hard sample，再给 terminal bonus。

### Hard constraint 与 soft reward 必须分开

Hard constraints：

- max relative power；
- max EVM；
- max PSD distance；
- max timing shift；
- operator bounds。

LLM 和 PPO 都不能改。

## 10. LLM Reward Evolution

LLM 不允许自由写 Python reward，只能使用 Reward DSL。

可选 primitives：

```text
Hardness:
  ce_loss
  logit_margin
  margin_delta
  confidence_drop

Diversity:
  none
  parameter_diversity
  embedding_diversity
  operator_usage_balance

Penalty:
  relative_power
  evm
  psd_distance
  action_energy
  saturation
  duplicate_rate

Schedule:
  constant
  linear
  two_stage
```

LLM 可以改：

- primitive selection；
- reward composition；
- weights；
- curriculum；
- terminal bonus。

LLM 不能改：

- RF hard constraints；
- data split；
- victim；
- PPO architecture；
- final test；
- physical operator range。

## 11. LLM 输入输出

输入固定为结构化 `ExperimentSummary`：

```yaml
reward: ...
ppo:
  mean_reward: ...
  entropy: ...
  approx_kl: ...
generation:
  valid_hard_rate: ...
  margin_drop: ...
  duplicate_rate: ...
operator_usage: ...
rf_validity: ...
quick_defense:
  clean_acc_delta: ...
  seen_robust_acc: ...
  holdout_robust_acc: ...
diagnostics: ...
```

输出固定 schema：

```yaml
hardness:
  primitive: margin_delta
  weight: 1.2
diversity:
  primitive: embedding_diversity
  weight: 0.2
penalties:
  relative_power: 0.3
  psd_distance: 0.6
schedule:
  type: two_stage
terminal_bonus:
  enabled: true
  weight: 0.8
```

## 12. 必须和传统 Reward Search 比

实验必须包含：

```text
PPO-Manual
PPO-RandomRewardSearch
PPO-TPE
PPO-LLM-OneShot
PPO-LLM-Evolve
```

只有当 LLM 获得更高 holdout robustness，或用更少 candidate evaluation 达到同等结果，才有资格成为主要创新点。

## 13. Defense Utility

Winner 不能按 fooling rate 直接选择。

必须：

```text
Reward
 ↓
PPO
 ↓
Hard Samples
 ↓
Quick Defender Training
 ↓
Holdout Robustness
```

先过滤：

\[
RFViolation\le\tau_v
\]

\[
\Delta CleanAcc\ge-\tau_c
\]

再最大化：

\[
RA_{\text{holdout}}
\]

## 14. 数据划分

至少逻辑拆成：

```text
Generator Train
Reward Validation
Defense Train
Final Test
```

Final Test 不得进入 PPO、LLM summary、early stopping 或 candidate selection。

## 15. Identity Preservation Gate

所有 operator 在进入 PPO 前必须做 sweep：

\[
D_{intra}=\|e(x')-\mu_y\|
\]

\[
D_{other}=\min_{j\ne y}\|e(x')-\mu_j\|
\]

并统计：

- clean-class retention；
- embedding drift；
- RF validity；
- preprocessing sensitivity。

如果极小变化就显著改变 class manifold，则禁止作为 label-preserving training operator。

## 16. Baselines

### PPO 是否有价值

```text
Random RF Search
CEM/CMA-ES
PPO-Fixed
```

PPO 的主要潜在价值是：

\[
\boxed{\text{amortized constrained search}}
\]

训练后面对新样本只需要少量 policy forward。

### 防御是否有价值

```text
Clean Training
Standard RF Augmentation
PGD-AT
PPO-Fixed
Ours
```

### LLM 是否有价值

```text
Manual
Random
TPE
LLM One-Shot
LLM Evolution
```

## 17. Go / No-Go Gates

### Gate 0：Operator

必须：

- realistic range 内有效；
- preprocessing 不会完全抹掉；
- identity preservation 尚可。

### Gate 1：PPO

相同 victim-query budget：

```text
PPO-Fixed > Random RF Search
```

至少在 valid hard rate、margin drop 或 amortized efficiency 中成立。

### Gate 2：PPO vs 低维优化器

和 CEM/CMA-ES 比。如果 PPO 不体现 reusable/state-conditioned/amortized 优势，重新审视其必要性。

### Gate 3：Defense Utility

```text
PPO hard-sample training > standard RF augmentation
```

最好在 hidden/cross-model robustness 上成立。

### Gate 4：LLM

比较 Manual / Random / TPE / LLM。若无性能或搜索效率优势，LLM 降级为辅助工具。

## 18. 推荐工程结构

```text
project/
├── configs/
├── rffi/
│   ├── data/
│   ├── victim/
│   ├── rfops/
│   ├── metrics/
│   ├── env/
│   ├── reward/
│   ├── ppo/
│   ├── outer_loop/
│   ├── defense/
│   └── baselines/
├── tools/
├── tests/
└── reports/
```

关键接口：

```python
VictimAdapter.preprocess()
VictimAdapter.logits()
VictimAdapter.embedding()

RFOperator.apply()
RFConstraint.check()

RewardEngine.compute()
ExperimentSummary.build()
```

## 19. Codex 执行顺序

```text
Task 1  审计现有 RFFI pipeline
Task 2  实现 VictimAdapter
Task 3  实现 Tier-A operators
Task 4  operator_sweep + identity preservation
Task 5  冻结 action range
Task 6  vectorized PPO env
Task 7  manual Reward DSL
Task 8  Random vs PPO-Fixed
Task 9  CEM/CMA-ES baseline
Task 10 defense utility
Task 11 TPE reward search
Task 12 LLM one-shot
Task 13 LLM evolution
```

Gate 1 前禁止投入完整 LLM outer-loop 工程。

## 20. 论文贡献可如何表述

如果实验成立，建议：

1. 提出 RF-identity-aware constrained hard-sample space；
2. 提出 state-conditioned PPO amortized hard-example generator；
3. 提出 defense-utility-guided reward evolution；
4. 系统比较 manual / random / TPE / LLM reward search。

不要写“首次用 LLM 生成 RFFI I/Q”，因为本方案中 LLM 并不生成 I/Q。

## 21. 优缺点

### 优点

- 最容易跑通；
- PPO action 低维；
- RF validity 可控；
- 身份语义容易审计；
- 计算成本可控；
- 适合硕士研究的保底主线。

### 缺点

- 与 AdvTG 的 PPO-tuned LLM generator 不同构；
- LLM 容易被质疑成 reward tuning；
- 生成空间由人工 operator 限定；
- 可能遗漏未知的 waveform-level hard modes。

## 22. 推荐定位

\[
\boxed{\text{低风险主线 / 保底方案}}
\]

如果生成式新版本的核心 feasibility gate 失败，则回退到本方案。

---

## 参考依据

1. Sun et al., *AdvTG: An Adversarial Traffic Generation Framework to Deceive DL-Based Malicious Traffic Detection Models*, The Web Conference 2025.  
   https://github.com/TrafficDetection-art/AdvTG

2. Ma et al., *Adversarial Attacks Against Deep Learning-Based Radio Frequency Fingerprint Identification*, IEEE TMC, DOI: 10.1109/TMC.2025.3646257.  
   https://livrepository.liverpool.ac.uk/3196119/

3. Guo et al., *Toward Open-Set Specific Emitter Identification Using Auxiliary Classifier Generative Adversarial Network and OpenMax*, IEEE TCCN, DOI: 10.1109/TCCN.2024.3408417.
