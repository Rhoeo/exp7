# 新调研版：RFF Generative Model + PPO 的 AdvTG 式迁移方案与可行性分析

> 版本日期：2026-08-27  
> 用途：研究决策 + Codex 后续技术实施。  
> 研究定位：**利用可生成 RFF/IQ 的领域生成模型，在防御研究中通过 PPO 挖掘受约束困难样本，并用于 RFFI 鲁棒训练。**  
> 本文比较 Diffusion、VQ-VAE+GPT、Continuous Transformer 与 RF-Operator PPO 四条路线。

---

## 1. 调研后的核心修正

此前不能把问题描述成：

> “没有模型可以生成射频指纹 I/Q 序列。”

这个判断不准确。

截至 2026-08，公开文献已经分别证明：

1. LLM/Transformer 可以用于 RFFI 识别与 RF 表征；
2. 生成模型可以直接生成时序 RF/IQ 信号；
3. GAN / diffusion 已进入 RFF/SEI 的样本生成、增强或 fingerprint imitation 场景；
4. PPO 已在 RFF-LLM 工作中出现，但用途是动态知识蒸馏温度；
5. RFFI 对抗样本已有 FGSM、PGD、UAP、GAN 等路线。

本轮检索中**没有发现公开工作明确完成**：

\[
\boxed{
\text{RFF/IQ generative LLM}
+
\text{PPO victim feedback}
+
\text{hard/adversarial IQ generation}
+
\text{defense training}
}
\]

因此潜在空白不是“IQ generation”，而是：

> **是否能构建一个类似 AdvTG 的 domain generative model，并通过 PPO 将其从“会生成 RF 信号”进一步优化成“会生成对 RFFI 防御训练有价值的困难 RF 信号”。**

---

# 2. LLM 用于 RFFI：已有明确先例

## 2.1 BERT-LightRFFI

Gao 等，Science China Information Sciences 2025：

**Let RFF do the talking: large language model enabled lightweight RFFI for 6G edge intelligence**

结构概念：

```text
unlabeled RF data
   ↓
BERT-style pretraining
   ↓
RFF feature extractor
   ↓
knowledge distillation
   ↓
BERT-Light
   ↓
few-shot RFFI
```

意义：

\[
\boxed{\text{Transformer/LLM-style representation learning 已直接进入 RFFI}}
\]

正式信息：

- Sci China Inf Sci, 2025, 68(7):170308
- DOI: 10.1007/s11432-024-4463-0

---

## 2.2 RFF-LLM：modified GPT-2 + PPO

Zheng 等，IEEE Wireless Communications Letters 2025：

**UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks**

结构：

```text
UAV I/Q
   ↓
modified GPT-2 RFF-LLM
   ↓
teacher
   ↓
knowledge distillation
   ↓
Lite-HRNet
```

其中 PPO 用于：

```text
dynamically adjust distillation temperature
```

而不是生成 adversarial IQ。

重要意义：

- GPT-2-style 模型用于 RFFI 已有正式先例；
- PPO 与 RFFI/LLM 同时出现已有先例；
- 但 PPO 尚未扮演“优化 RFF waveform generator”的角色。

论文信息：

- IEEE Wireless Communications Letters, 14(11), 3769–3773
- DOI: 10.1109/LWC.2025.3603423

---

## 2.3 Shapelet + LLM

Zhao 等，2026：

**Generalizable and Interpretable RF Fingerprinting with Shapelet-Enhanced Large Language Models**

将：

- variable-length 2D shapelets；
- pretrained LLM；

结合用于 RF fingerprinting，强调：

- local I/Q temporal pattern；
- long-range dependency；
- cross-domain generalization；
- few-shot prototype generation。

但这里的 prototype generation：

\[
\neq
\]

raw IQ waveform generation。

---

# 3. RF/IQ 生成模型：已经存在

## 3.1 RF-Diffusion

Chi 等，MobiCom 2024：

**RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion**

核心就是直接生成：

\[
\boxed{\text{time-series RF data}}
\]

方法包含：

- Time-Frequency Diffusion；
- complex-valued RF modeling；
- Hierarchical Diffusion Transformer。

已展示 Wi-Fi 和 FMCW 信号合成。

因此：

\[
\boxed{\text{直接生成高质量 RF/IQ 时序在方法上已经被证明可行}}
\]

但 RF-Diffusion 并非专门为“保留某个 transmitter fingerprint”设计。

---

# 4. RFF/SEI 生成式模型先例

## 4.1 GAN 模仿 RF fingerprint

Xu 等，ICC 2022：

**Colluding RF Fingerprint Impersonation Attack Based on Generative Adversarial Network**

使用 GAN generator 学习输出具有目标合法用户相似 RFF 的信号。

意义：

> “生成模型可以对 RF fingerprint 特性进行目标化建模”已有直接先例。

所以不能再说 RFF/IQ 只能被“扰动”，不能被“生成”。

---

## 4.2 ACGAN 生成 outlier 用于开放集 SEI

Guo 等，IEEE TCCN 2024：

**Toward Open-Set Specific Emitter Identification Using Auxiliary Classifier Generative Adversarial Network and OpenMax**

结构：

```text
ACGAN
 ↓
outlier samples
 ↓
OpenMax
 ↓
unknown emitter rejection
```

这与本研究的**防御导向生成**非常接近：

> 合成特殊边界/异常样本，帮助系统学习未知空间，而不是把攻击成功本身当最终目标。

---

## 4.3 Diffusion 用于 SEI synthetic augmentation

2026：

**Adversarial diffusion synthesis for specific emitter identification: multi-scale signal augmentation with cross-domain constraints**

该工作使用 diffusion 进行 SEI synthetic augmentation，并显式考虑：

- time-domain waveform；
- frequency-domain modulation；
- sequential features；
- physical/statistical/semantic consistency。

这说明：

\[
\boxed{\text{生成 RF signal + 领域一致性约束 + SEI 性能提升}}
\]

已经是现实研究路线。

因此我们不能把“用 diffusion/GAN 合成 SEI 样本”本身作为主要创新。

---

# 5. RFFI adversarial example 现状

Ma 等的 RFFI 工作系统研究：

```text
FGSM
PGD
UAP
```

并在：

```text
CNN
LSTM
GRU
```

上验证，也考虑 practical wireless context。

因此本项目若做 defense-oriented hard generation，至少应有：

```text
FGSM / PGD / UAP
```

作为基础参考。

---

# 6. 与 AdvTG 的真正同构映射

AdvTG：

```text
HTTP traffic
    ↓
Domain LLM
    ↓ copy
Frozen reference LLM      PPO-tuned LLM
          \                  /
           \---- KL --------/
                  ↑
              detector reward
                  ↑
          generated traffic
```

若做最忠实的 RFFI 映射，应是：

```text
RFF/IQ
  ↓
Domain RF Generator
  ↓ copy
Frozen reference generator      PPO-tuned generator
              \                    /
               \--- consistency --/
                         ↑
                   victim feedback
                         ↑
                 generated hard IQ
```

对应：

| AdvTG | RFFI 新版 |
|---|---|
| HTTP tokens | RF latent/tokens / continuous RF patches |
| Domain LLM | Domain RF generative model |
| LLM output | RF waveform / RF tokens |
| PPO fine-tunes LLM | PPO fine-tunes RF generator |
| KL to SFT model | policy KL / latent consistency / waveform & fingerprint preservation |
| functional fields | identity-preserving + communication-valid constraints |
| detector reward | RFFI hardness + RF/identity constraints |
| final evasion | defense-oriented hard-sample training |

---

# 7. 真正的难点：不是生成 IQ，而是保留 transmitter identity

普通生成目标：

\[
p_G(x)\approx p_{data}(x)
\]

RFFI 需要：

\[
p_G(x|y_{\text{device}})
\]

而且生成信号必须保留设备 y 的微弱硬件特征。

单纯：

\[
L_{time}=\|x-\hat x\|
\]

不够，因为微弱 fingerprint 可能被 reconstruction 模型平滑掉。

建议生成/重构模型至少使用：

\[
L=
\lambda_tL_{time}
+\lambda_fL_{freq}
+\lambda_{rff}L_{fingerprint}
+\lambda_cL_{class}
\]

其中使用 frozen RFF encoders：

\[
L_{fingerprint}
=
d(f_A(x),f_A(\hat x))
+
d(f_B(x),f_B(\hat x))
\]

并使用两个不同 architecture 的 evaluator，避免 generator 只适配单个 classifier。

---

# 8. 四条候选路线

## 路线 A：RF-Operator PPO

```text
clean IQ
 ↓
PPO
 ↓
low-dimensional RF parameters
 ↓
deterministic transformation
 ↓
hard IQ
```

### 优点

- 实现最简单；
- 物理可解释；
- PPO 最容易训练；
- constraint 最容易控制。

### 缺点

- 生成自由度低；
- 依赖人工 operator；
- LLM 不是 generator；
- 与 AdvTG 不同构。

### 可行性

\[
\boxed{\text{高}}
\]

定位：保底 + baseline。

---

## 路线 B：Conditional RF Diffusion + RL/PPO 控制

结构：

```text
device condition y
       +
reference / latent
       ↓
conditional RF diffusion
       ↓
synthetic IQ
       ↓
victim + identity + RF metrics
```

### 为什么有基础

RF-Diffusion 已证明复杂 RF time-series generation 可行；2026 SEI diffusion 工作又证明生成式 augmentation + cross-domain constraints 可用于 emitter identification。

### PPO 的三种接法

#### B1. PPO 控制 latent/condition

PPO action 控制：

```text
guidance scale
latent direction
conditioning vector
noise/seed embedding
selected denoising controls
```

优点：较稳定。  
缺点：不是 PPO 直接微调完整 generator。

#### B2. RL fine-tune diffusion policy

最接近“优化 generator”，但工程风险很高。

#### B3. Diffusion 固定，PPO 搜 latent

\[
z_{t+1}=z_t+a_t
\]

\[
x'=G_{diff}(z_t,y)
\]

这是最现实的 diffusion + PPO 方式。

### 主要风险

- diffusion sampling 成本高；
- PPO episode 多次调用 generator；
- victim forward 与 diffusion forward 双重开销；
- RL credit assignment 更麻烦。

### 可行性

\[
\boxed{\text{中高（生成） / 中低（PPO 直接微调）}}
\]

---

## 路线 C：VQ-VAE + GPT / Transformer

这是**与 AdvTG 最同构**的一条路线。

### C1. IQ tokenization

\[
x\xrightarrow{Encoder}z
\]

\[
z\xrightarrow{VQ}[q_1,q_2,\dots,q_N]
\]

Decoder：

\[
[q_1,\dots,q_N]\rightarrow\hat x
\]

即：

```text
continuous IQ
 ↓
discrete RF tokens
```

### C2. Domain RF-GPT

训练：

\[
p(q_t|q_{<t},y_{\text{device}},c)
\]

条件可包含：

```text
device ID
protocol
domain/capture
SNR
```

生成：

```text
device y
 ↓
RF-GPT
 ↓
RF tokens
 ↓
VQ decoder
 ↓
IQ
```

### C3. AdvTG 式 PPO

```text
Domain RF-GPT
      ↓ copy
┌──────────────┐
│              │
Frozen       PPO-tuned
reference      GPT
│              │
└──── KL ──────┘
```

reward：

\[
R=
\lambda_hR_{hard}
-\lambda_{KL}D_{KL}
-\lambda_{RF}V_{RF}
-\lambda_{ID}V_{ID}
+\lambda_DDiversity
\]

### 优点

- HTTP token ↔ RF token，映射清晰；
- PPO 的 action 是 next RF token；
- reference-policy KL 自然；
- LLM/Transformer 是 generator 本体，而不是外挂；
- 论文技术逻辑最完整。

### 最大风险

#### 1. VQ tokenizer 抹掉 fingerprint

这是最致命的风险。

#### 2. 数据量

RFFI 数据通常远小于 NLP，因此应该做：

```text
small RF-GPT / autoregressive RF Transformer
```

而不是追求真正的大参数通用 LLM。

#### 3. PPO 稳定性

sequence-level reward、KL、generation cost 都可能导致训练困难。

### 可行性

\[
\boxed{\text{中}}
\]

科研潜力最高，工程风险最高。

---

## 路线 D：Continuous GPT / Transformer

不做 VQ。

将 IQ 切成连续 patch：

\[
x\rightarrow p_1,\dots,p_N
\]

Transformer：

\[
p(p_t|p_{<t})
\]

输出 continuous distribution parameters，例如：

\[
(\mu_t,\sigma_t)
\]

### 优点

- 无 quantization artifact；
- Transformer 仍为生成器。

### 缺点

- PPO continuous policy 更复杂；
- policy KL 不如离散 token 简洁；
- generation error accumulation；
- likelihood calibration 难；
- 工程成熟度低于 VQ-GPT。

### 可行性

\[
\boxed{\text{中低}}
\]

不建议首选。

---

# 9. 四路线对比

| 维度 | RF-Operator PPO | Diffusion | VQ-VAE+GPT | Continuous GPT |
|---|---:|---:|---:|---:|
| 实现难度 | 低 | 中高 | 高 | 高 |
| GPU 成本 | 低 | 高 | 中高 | 中高 |
| 与 AdvTG 同构 | 低 | 中 | **最高** | 高 |
| 直接生成 IQ | 间接 | **是** | **是** | **是** |
| 身份约束可控 | 高 | 中 | 中 | 中 |
| 生成自由度 | 低 | 高 | 高 | 高 |
| PPO 易接入 | 高 | 中低 | **中高** | 中 |
| 复现风险 | 低 | 中 | 高 | 高 |
| Transformer/LLM 角色自然 | 低 | 中 | **高** | 高 |
| 创新潜力 | 中 | 高 | **最高** | 高 |
| 硕士时间风险 | 低 | 中高 | **高** | 高 |

---

# 10. 不应该直接选四选一：采用 feasibility funnel

真正的第一问题不是：

> PPO 怎么训？

而是：

\[
\boxed{\text{Base RF generative model 能否保留 device fingerprint？}}
\]

如果这一步失败，PPO 没有意义。

---

# 11. Stage G0：共同 RFFI 基线

冻结：

```text
dataset
frame extraction
normalization
train/val/test
Victim A
Evaluator B
RF metrics
```

至少两个不同 architecture：

```text
Victim A
Independent Evaluator B
```

作用：

> 防止 generator 学会只骗某一个 classifier 的 feature space。

---

# 12. Stage G1：只做生成，不做 PPO

候选：

```text
VQ-VAE
conditional GAN/VAE baseline
conditional diffusion
```

回答：

> 能不能生成/重构“属于指定设备”的有效 IQ？

## Reconstruction metrics

- NMSE；
- waveform correlation；
- PSD distance；
- EVM-like；
- power statistics。

## Fingerprint preservation

\[
Acc_A(\hat x),\quad Acc_B(\hat x)
\]

\[
d(f_A(x),f_A(\hat x))
\]

\[
d(f_B(x),f_B(\hat x))
\]

## Conditional generation

给定 device y：

\[
x_G\sim p_G(x|y)
\]

统计：

\[
P_A(\hat y=y),\quad P_B(\hat y=y)
\]

同时做：

- diversity；
- nearest-neighbor；
- memorization/duplicate checks。

---

# 13. G1 Go / No-Go

Generator 至少需要：

1. reconstructed signals 在 A/B 上均有较高 identity retention；
2. synthetic samples 不是训练样本复制；
3. waveform / PSD / power 合理；
4. conditional-by-device 生成具有可分辨性。

若 fingerprint retention 明显失败：

\[
\boxed{\text{停止该 generative route}}
\]

如果 VQ 失败，可尝试 diffusion；若两者都失败，回保守版。

---

# 14. Stage G2：优先验证 VQ tokenizer

不要直接训练 GPT。

先做：

\[
IQ\rightarrow VQ\ tokens\rightarrow IQ
\]

这是 VQ-GPT 最大的单点风险。

建议 loss：

\[
L_{VQ}
=
\lambda_{rec}L_{rec}
+\lambda_{fft}L_{fft}
+\lambda_{rff}L_{rff}
+\lambda_{cls}L_{cls}
+L_{commit}
\]

其中：

\[
L_{rff}
=
d(f_A(x),f_A(\hat x))
+
d(f_B(x),f_B(\hat x))
\]

\[
L_{cls}
=
CE(g_A(\hat x),y)
+
CE(g_B(\hat x),y)
\]

若 VQ reconstruction 在 cross-model identity 上明显失败：

```text
STOP VQ-GPT
```

不要继续浪费算力。

---

# 15. Stage G3：Domain RF-GPT

若 tokenizer 通过，再训练小型 autoregressive Transformer。

建议先从：

```text
4~8 Transformer blocks
d_model = 256~512
```

开始。

不是 7B，也不需要追求“大模型规模”。

训练：

\[
\max_\theta\sum_t\log p_\theta(q_t|q_{<t},y)
\]

验证：

- conditional device fidelity；
- RF validity；
- diversity；
- memorization；
- unseen capture generation。

---

# 16. Stage G4：PPO 前先做 Hardness Probe

改变：

```text
temperature
top-p
token perturbation
condition interpolation
```

检查 base generator 是否自然产生：

```text
easy / medium / hard
```

样本。

如果所有 synthetic samples 都是 trivial easy，PPO 需要跨越的生成分布距离过大，风险会明显增加。

---

# 17. Stage G5：AdvTG 式 PPO

冻结 reference：

\[
\pi_{ref}=\pi_{SFT}
\]

训练 policy：

\[
\pi_\phi,\quad \phi\leftarrow\theta_{SFT}
\]

如果 VQ-GPT：

```text
state  = RF-token prefix + device condition
action = next RF token
```

这与文本 LLM PPO 最同构。

---

# 18. PPO Reward

研究目的是 defense，因此 reward 不应只有 fooling。

\[
R=
\lambda_hR_{hard}
-\lambda_{KL}D_{KL}(\pi_\phi||\pi_{ref})
-\lambda_{RF}V_{RF}
-\lambda_{ID}V_{ID}
+\lambda_DDiversity
\]

### Hardness

\[
R_{hard}
=
\max_{j\neq y}z_j-z_y
\]

### Reference KL

\[
D_{KL}(\pi_\phi||\pi_{ref})
\]

约束生成策略不要偏离 base RF language model 太远。

### RF validity

Decoded IQ 检查：

- power；
- PSD；
- EVM-like；
- waveform statistics；
- protocol-specific validity（如果可计算）。

### Identity preservation

核心不是“保证 victim A 仍预测 y”，因为 victim A 同时是 hardness target。

应使用独立 Evaluator B 或 identity manifold：

\[
V_{ID}
=
\max(0,d_B(\hat x,\mu_y)-\tau_{id})
\]

---

# 19. 方法的关键语义

目标不是：

\[
\text{把设备 A 的波形生成成设备 B}
\]

而是：

\[
\boxed{\text{在设备 A 的身份流形附近寻找分类器最难处理的合法/近合法样本}}
\]

形式化：

\[
x_{hard}\in\mathcal M_y
\]

同时：

\[
Margin_{victim}(x_{hard})\downarrow
\]

这一区分对于“防御导向困难样本生成”非常重要。

---

# 20. Defense Utility 才是最终评价

即使 PPO hard rate 很高，也不能直接宣布方法有效。

必须：

```text
PPO RF-GPT
 ↓
hard IQ pool
 ↓
defender training
 ↓
hidden perturbation / unseen model
 ↓
robustness
```

最终主目标：

\[
\boxed{\Delta RobustAcc_{\text{hidden}}}
\]

而不是 ASR。

---

# 21. 新版本是否还需要 LLM Reward Evolution？

**不建议第一版同时加。**

如果采用 VQ-GPT：

\[
\boxed{\text{Transformer/GPT 已经是 generator/policy backbone}}
\]

这时再加一个外层 LLM 调 reward，会造成方法堆叠。

新版本 V1 推荐：

```text
VQ-VAE
+
RF-GPT
+
PPO
+
fixed RF/identity-aware reward
```

只有 reward balancing 被实验确认成主要瓶颈时，再增加 outer LLM reward evolution。

因此两版本的 LLM 角色：

| 版本 | LLM/Transformer 角色 |
|---|---|
| 保守版 | 外层 Reward Designer |
| 新版 | **RF sequence generator / policy backbone** |
| 新版增强 | 可选再加 Reward Designer |

---

# 22. 为什么新版更能回答“为什么用 LLM？”

技术链：

\[
IQ
\rightarrow
RF\ tokens
\rightarrow
GPT
\rightarrow
PPO
\]

解释：

1. VQ tokenizer 把 continuous I/Q 映射为离散 RF token sequence；
2. GPT-style autoregressive model 学习 token 长程依赖和 device-conditioned generation；
3. PPO 像 RLHF 一样，在 reference-policy KL 下调整生成分布；
4. decoded waveform 由 RF/identity constraints 限制。

因此 LLM/Transformer 不是为了“AI 味”，而是生成策略本体。

---

# 23. 风险分析

## R1. VQ reconstruction 破坏 fingerprint

概率：高  
影响：致命

缓解：

- 增大 codebook；
- 减少 downsampling；
- multi-scale encoder；
- RFF perceptual loss；
- multi-model preservation loss。

Gate：

```text
cross-model reconstructed RFF retention fails
→ stop VQ-GPT
```

---

## R2. GPT 数据不足

概率：中高  
影响：高

缓解：

- 小模型；
- 共享多设备 token vocabulary；
- patch/token reduction；
- self-supervised pretraining；
- 多 capture/domain 联合训练。

不要为了“LLM”而盲目增加参数量。

---

## R3. Generator memorization

检查：

- nearest-neighbor distance；
- train-vs-generated similarity；
- duplicate rate；
- holdout capture diversity。

---

## R4. PPO reward hacking

典型现象：

```text
Victim A hardness ↑
but
Evaluator B identity retention ↓
```

缓解：

- frozen reference KL；
- independent identity evaluator；
- RF validity；
- hidden model；
- diversity checks。

---

## R5. PPO 计算成本

Diffusion 路线尤其严重。

VQ-GPT 的优势是：

> autoregressive discrete policy + PPO 与成熟 RLHF 路径最相似。

因此若目标是“AdvTG 式 PPO”，VQ-GPT 比 diffusion 更自然。

---

## R6. 被质疑“只是文本 token 换 RF token”

必须加入真正 RFFI-specific 技术：

- fingerprint-preserving tokenizer；
- identity-manifold constraint；
- dual-model identity validation；
- RF + RFF consistency reward；
- defense-utility protocol。

否则创新性不足。

---

# 24. 资源可行性

## 保守版

计算主要在 victim forward，单 GPU 容易实施。

## VQ-VAE + GPT

如果 frame 只有数百 I/Q samples，不需要真正的大模型。

可从：

```text
VQ encoder/decoder: few million params
RF-GPT: 10M~100M class
```

做概念验证。

推荐术语：

```text
GPT-style RF generative model
RF language model
autoregressive RF Transformer
```

除非参数/预训练规模足够，否则不要强行声称是通用“大语言模型”。

---

# 25. 推荐实验路线

## Phase 0

建立：

- clean victim A；
- evaluator B；
- PGD/UAP；
- standard RF augmentation。

## Phase 1

只实现 VQ-VAE。

回答：

> tokenization/reconstruction 是否保留 fingerprint？

## Phase 2

若通过，训练 Domain RF-GPT。

回答：

> 能否 conditional 生成指定设备的 IQ？

## Phase 3

若通过，加入 reference + PPO。

回答：

> PPO 是否能在 KL + identity constraint 下提高 valid hard-sample rate？

## Phase 4

做 defense training。

回答：

> 是否提高 unseen robustness？

## Phase 5

与保守版比较：

```text
RF-Operator PPO
vs
RF-GPT PPO
```

---

# 26. Go / No-Go 决策树

```text
START
  │
  ▼
VQ-VAE reconstruction
  │
  ├── fingerprint retention FAIL
  │       ↓
  │   conditional diffusion
  │       │
  │       ├── FAIL → 回保守版
  │       └── PASS → diffusion route
  │
  └── PASS
          ↓
      Train RF-GPT
          │
          ├── conditional generation FAIL
          │       ↓
          │   diffusion / conservative
          │
          └── PASS
                  ↓
             PPO fine-tuning
                  │
                  ├── unstable/no benefit
                  │       ↓
                  │   latent search / conservative
                  │
                  └── PASS
                          ↓
                    defense training
                          │
                          ├── hidden gain = no → hypothesis fails
                          └── hidden gain = yes → full experiment
```

---

# 27. 两版本应共享工程底座

```text
common RFFI core
├── data
├── victim
├── RF metrics
├── identity metrics
└── defense evaluation

generator backend
├── operator_policy
├── vq_gpt
└── diffusion
```

这样新路线失败也不会浪费全部工程工作。

---

# 28. 推荐 Codex 目录

```text
project/
├── rffi_core/
│   ├── data/
│   ├── victim/
│   ├── metrics/
│   ├── identity/
│   └── defense/
├── generators/
│   ├── operator_policy/
│   ├── vqvae/
│   ├── rf_gpt/
│   └── diffusion/
├── rl/
│   ├── ppo/
│   ├── reward/
│   └── reference_policy/
├── baselines/
│   ├── fgsm/
│   ├── pgd/
│   ├── uap/
│   └── standard_aug/
├── tools/
└── reports/
```

---

# 29. Codex 第一阶段严禁直接做 PPO

执行顺序：

```text
Task 1
审计现有数据：
- IQ shape
- frame length
- normalization
- devices
- captures
- domains
- current victim

Task 2
训练/加载 Victim A + Independent Evaluator B。

Task 3
实现 fingerprint-preservation metrics。

Task 4
实现 VQ-VAE baseline。

Task 5
只做 reconstruction + RFF preservation evaluation。

Task 6
Gate 通过后训练 conditional RF-GPT。

Task 7
验证 conditional generation / diversity / memorization。

Task 8
Gate 通过后复制 reference + PPO policy。

Task 9
实现 identity-aware PPO reward。

Task 10
最后做 defender training。
```

---

# 30. 如果 VQ-GPT 成功，论文贡献可组织为

### Contribution 1：Fingerprint-Preserving RF Tokenizer

将 raw complex I/Q 映射成离散 RF tokens，同时通过 waveform/frequency/RFF consistency 保留设备身份。

### Contribution 2：PPO-Tuned RF Language Generator

在 reference-policy KL 下，根据 RFFI victim feedback 调整 conditional RF generation policy。

### Contribution 3：Identity-Constrained Defense-Oriented Hard Samples

不是让波形冒充另一设备，而是在原设备 identity manifold 附近挖掘分类器困难区域，并用 unseen defense gain 评价其价值。

这比“LLM 调 reward”更接近一个完整的新方法。

---

# 31. 最终推荐

当前建议优先级：

\[
\boxed{1.\ VQ\text{-}VAE+GPT\ feasibility}
\]

\[
\boxed{2.\ Conditional\ Diffusion\ 作为生成质量备选}
\]

\[
\boxed{3.\ RF\text{-}Operator\ PPO\ 作为保底和 baseline}
\]

原因：

- VQ-GPT 与 AdvTG 最同构；
- PPO 接离散 autoregressive policy 最自然；
- reference-policy KL 容易定义；
- 比 diffusion-PPO 更接近成熟 RLHF 工程路径；
- tokenizer 一旦失败可以很早止损；
- 保守版共用 victim/metrics/defense pipeline。

因此真正应该首先验证的不是 PPO，而是：

\[
\boxed{\text{RFF fingerprint 能否在 VQ tokenization / reconstruction 后可靠保留}}
\]

这是新版本最关键的单点可行性问题。

---

# 32. 关键文献

1. Sun, P. et al. **AdvTG: An Adversarial Traffic Generation Framework to Deceive DL-Based Malicious Traffic Detection Models**. The Web Conference 2025.  
   https://github.com/TrafficDetection-art/AdvTG

2. Gao, N., Liu, Y., Zhang, Q. F., Li, X., Jin, S. **Let RFF do the talking: large language model enabled lightweight RFFI for 6G edge intelligence**. Science China Information Sciences, 2025, 68(7):170308. DOI: 10.1007/s11432-024-4463-0.  
   https://scis.scichina.com/scis-commun.html

3. Zheng, H., Gao, N., Cai, D., Jin, S., Matthaiou, M. **UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks**. IEEE Wireless Communications Letters, 2025, 14(11):3769-3773. DOI: 10.1109/LWC.2025.3603423.  
   https://arxiv.org/abs/2508.12597

4. Zhao, T., Zhang, J., Xu, H., Sun, X., Dai, J., Wang, X. **Generalizable and Interpretable RF Fingerprinting with Shapelet-Enhanced Large Language Models**. arXiv:2602.03035, 2026.  
   https://arxiv.org/abs/2602.03035

5. Chi, G. et al. **RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion**. ACM MobiCom 2024.  
   https://arxiv.org/abs/2404.09140

6. Xu, Y., Liu, M., Peng, L., Zhang, J., Zheng, Y. **Colluding RF Fingerprint Impersonation Attack Based on Generative Adversarial Network**. IEEE ICC 2022. DOI: 10.1109/ICC45855.2022.9838574.  
   https://livrepository.liverpool.ac.uk/3148337/

7. Guo, L., Liu, C., Liu, Y., Lin, Y., Gui, G. **Toward Open-Set Specific Emitter Identification Using Auxiliary Classifier Generative Adversarial Network and OpenMax**. IEEE Transactions on Cognitive Communications and Networking, 2024, 10(6):2019-2028. DOI: 10.1109/TCCN.2024.3408417.

8. Duan, Y. et al. **Adversarial diffusion synthesis for specific emitter identification: multi-scale signal augmentation with cross-domain constraints**. Digital Communications and Networks, 2026. DOI: 10.1016/j.dcan.2026.07.006.  
   https://www.sciencedirect.com/science/article/pii/S235286482600091X

9. Ma, J., Zhang, J., Shen, G., Marshall, A., Chang, C.-H. **Adversarial Attacks Against Deep Learning-Based Radio Frequency Fingerprint Identification**. IEEE Transactions on Mobile Computing, 2026. DOI: 10.1109/TMC.2025.3646257.  
   https://livrepository.liverpool.ac.uk/3196119/

10. Liu, C. et al. **Overcoming Data Limitations: A Few-Shot Specific Emitter Identification Method Using Self-Supervised Learning and Adversarial Augmentation**. IEEE Transactions on Information Forensics and Security, 2024, 19:500-513. DOI: 10.1109/TIFS.2023.3324394.
