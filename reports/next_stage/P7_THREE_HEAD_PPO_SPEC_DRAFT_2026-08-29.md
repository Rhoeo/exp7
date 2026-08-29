# P7 三头 PPO Token Editor 规范草案

日期：2026-08-29  
状态：**DRAFT — REQUIRES EXPERT APPROVAL**  
实施授权：**false**  
训练授权：**false**

本文件只定义可供专家逐项评审的技术规范。它不是实现许可，不包含 PPO 代码，也不得被解释为允许训练、读取 `final_test` 或使用 Evaluator B 参与策略闭环。

## 1. 启动依据与结论边界

P6 在固定 untouched-buffer Policy Gate 上满足项目 Gate Q：

- Greedy Q=256 conditional valid-hard rate 为 14.81%；query-matched random 为 6.55%。
- paired difference 为 +8.25 pp，95% CI 为 [5.83, 10.92] pp，Gate Q1 PASS。
- Greedy Q=128 达到 8.98%，超过 random Q=256 的 6.55%，查询预算为其 50%，Gate Q2 PASS。
- P6 所有正式条件下 Evaluator B retention 与数字域 RF-valid fraction 均为 100%。

上述结果只证明 Victim A score 对当前离散候选动作存在可利用信号。它不证明 PPO 一定优于 Greedy，不证明全分布攻击率，也不证明空口可实现性。

## 2. P7 研究问题

P7 只回答一个问题：能否把 P6 中昂贵的在线候选查询搜索，摊销成一个在每次编辑只查询一次 Victim A 的分层策略，同时保持候选合理性、数字域 RF 合法性和离线身份审计结果。

P7 不做以下工作：

- 不重新训练 tokenizer、VQ-VAE、Victim A 或 masked infill。
- 不把 causal RF-GPT 恢复为自由生成器。
- 不把 Evaluator B/C 放入 observation、reward、action mask、候选筛选、early stop 或 checkpoint selection。
- 不读取 `final_test`。
- 不声称 physically realizable 或 over-the-air feasible。
- 不训练 defender；defense utility 属于 P8。

## 3. 威胁模型

攻击类型为 untargeted score-based black-box hard-sample generation。

允许：

- 输入真实参考 IQ 的冻结 VQ token 序列和真实设备标签 (y)。
- 查询 Victim A 的 logits/confidence，并计算 true-class margin。
- 使用冻结 token graph、masked infill、VQ decoder 和公共 RF 投影。

禁止：

- 访问 Victim A 参数、梯度或中间 embedding。
- 对 Victim A、VQ decoder 或 RF 投影反向传播。
- 在训练闭环中读取 Evaluator B/C 输出。
- 依据 Policy Gate 或 `final_test` 结果调 reward、阈值、网络宽度或 PPO 超参数。

## 4. 冻结环境

P7 获批后仍必须复用以下 P6 口径：

- codebook size：1024；reference sequence length：2048。
- candidate graph：`artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz`。
- masked infill checkpoint：`runs/next_stage/p3_infill_hybrid_v2/best.pt`，只读冻结。
- candidate rule：latent/transition neighbor ∩ infill relative-probability ∩ actual decoded local RF precheck。
- ΔLL：4.0；candidate graph top-k：16；每位置最多保留 4 个 replacement。
- 无 Victim 的位置预排序池：最多 64 个位置。
- edit budget：最多 8 个不同位置；同一位置不重复编辑。
- 公共顺序：token edit → decode → RF projection → metric audit → Victim A query。
- projection：35 MHz sample rate、22 MHz occupied bandwidth、minimum SNR 22 dB、normalized peak delta ≤1.0、reference power match、perturbation bandlimit。

P7-v1 的候选集合从 clean reference 计算后在整个 episode 内冻结，以和 P6 保持同一动作空间。若以后改为每步重新计算受影响局部 infill context，必须作为新版本，同时重跑 random、Greedy 和 PPO；不得只让 PPO 使用动态候选。

## 5. 为什么是三个 actor head

PPO actor 的动作被明确拆为：

1. **Stop head**：是否终止 episode。
2. **Position head**：继续时，从合法且未编辑的位置中选择一个位置。
3. **Replacement head**：在该位置的 plausible candidate set 中选择替代 token。

PPO 还需要一个 scalar value head (V_psi(s_t)) 作为 critic。它不是动作分布，因此不计入“三头”。完整网络是“三个 actor head + 一个 critic value head”。

## 6. 分层动作概率

Stop head 输出 Bernoulli：

\[
p_{\mathrm{stop}}(s_t)=\sigma(f_{\mathrm{stop}}(g_t)).
\]

若采样 STOP，则 episode 终止且不产生新的 Victim 查询。若采样 CONTINUE：

\[
i_t\sim\pi_{\mathrm{pos}}(i\mid s_t),
\qquad
j_t\sim\pi_{\mathrm{rep}}(j\mid i_t,s_t).
\]

一次编辑动作的联合 log-probability 为：

\[
\log \pi_\phi(a_t\mid s_t)
=\log(1-p_{\mathrm{stop}})
+\log\pi_{\mathrm{pos}}(i_t\mid s_t)
+\log\pi_{\mathrm{rep}}(j_t\mid i_t,s_t).
\]

STOP 的 log-probability 为 \(\log p_{\mathrm{stop}}\)。PPO probability ratio 使用上述完整联合 log-probability，不得只对 replacement ratio 做 clipped objective。

## 7. Position head

每个合法位置 (i) 构造 reference feature：

- masked-infill entropy；
- candidate uncertainty/eligible candidate count；
- bidirectional token surprisal；
- candidate latent spread；
- decoder local sensitivity；
- global normalized position；
- 当前 edit mask 和剩余预算。

reference position score 草案为：

\[
u_i^{\mathrm{ref}}
=w_H z(H_i)+w_S z(S_i)+w_C z(\log(1+|C_i|))
+w_D z(D_i)+w_L z(L_i).
\]

其中 P6 已使用的初始权重为 (w_S=1.0,w_D=0.25,w_L=0.25)。新增的 entropy/candidate-uncertainty 权重和 temperature 必须经专家批准，并只允许在 `generator_search_dev` 上锁定。

\[
\pi_{\mathrm{pos}}^{\mathrm{ref}}(i\mid s)
=\operatorname{softmax}(u_i^{\mathrm{ref}}/\tau_{\mathrm{pos}}).
\]

Position policy 使用 residual parameterization：

\[
\ell_i^{\mathrm{pos}}
=\log \pi_{\mathrm{pos}}^{\mathrm{ref}}(i\mid s)
+f_{\mathrm{pos},\phi}(h_i,g_t).
\]

residual 最后一层零初始化，使训练开始时 position distribution 等于 reference prior。Position head **不加 KL penalty**；只报告 entropy、top-position concentration 和相对 reference 的诊断性 JS divergence。

## 8. Replacement head 与唯一的 KL

对选定位置 (i)，候选集合为 (C_i\)，大小 1–4。冻结 hybrid infill 给出 Transformer logits 加 transition-prior score：

\[
L_{ij}^{\mathrm{ref}}
=L_{ij}^{\mathrm{infill}}+w_TL_{ij}^{\mathrm{transition}}.
\]

reference replacement distribution 只在 (C_i) 上归一化：

\[
\pi_{\mathrm{rep}}^{\mathrm{ref}}(j\mid i,s)
=\operatorname{softmax}_{j\in C_i}(L_{ij}^{\mathrm{ref}}/\tau_{\mathrm{rep}}).
\]

policy 同样采用零初始化 residual：

\[
\ell_{ij}^{\mathrm{rep}}
=\log\pi_{\mathrm{rep}}^{\mathrm{ref}}(j\mid i,s)
+f_{\mathrm{rep},\phi}(h_i,e_j,g_t).
\]

只有 Replacement head 定义 reference KL：

\[
D_{\mathrm{KL}}^{\mathrm{rep}}
=D_{\mathrm{KL}}\left(
\pi_{\mathrm{rep},\phi}(\cdot\mid i,s)
\parallel
\pi_{\mathrm{rep}}^{\mathrm{ref}}(\cdot\mid i,s)
\right).
\]

不得把 Stop 或 Position head 强行与 masked-infill 做 KL，因为 masked-infill 没有定义 stop distribution，也没有天然定义完整 position distribution。

## 9. Stop head

Stop head 独立建模，不共享任何伪造的 masked-infill reference distribution，也不施加 reference KL。输入 (g_t) 至少包括：

- 当前 Victim margin 与相对 clean margin reduction；
- 已使用 edit/query 数；
- remaining budgets；
- 当前数字域 RF-valid summary；
- 可行动作数量和 position entropy；
- step index、是否已经 fool Victim A。

强制终止条件：

- Victim A 已被误分类且 RF-valid；
- edit budget 或 query budget 耗尽；
- 没有合法 position/replacement；
- 数值异常或连续 invalid action 达到上限。

除强制终止外，Stop head 可提前 no-op/early-stop。其学习信号来自 query/edit cost、终止收益和 PPO advantage，而不是 KL。

## 10. Observation/state

状态只包含黑盒攻击者可获得的信息：

- clean reference tokens、current edited tokens 和 edit mask；
- top-64 position table 及每位置最多 4 个 candidate feature；
- frozen infill logits/entropy、transition score、latent distance、decoder sensitivity；
- 当前和历史 Victim A logits 的压缩统计、true-class margin、margin delta；
- 当前投影后 RF metrics；
- action history；
- remaining edit/query budget 与 step index；
- known source device label embedding。

禁止加入：Victim gradients、Victim hidden features、Evaluator B/C output、Policy Gate outcome、`final_test` 信息。

## 11. Action mask

在 softmax 前以 `-inf` mask 非法动作：

- 已编辑位置；
- candidate set 为空的位置；
- 原 token 自身；
- 不在 token graph top-k 的 replacement；
- 未通过 ΔLL=4 relative probability 的 replacement；
- 未通过 actual-decoded local RF precheck 的 replacement；
- 会超过 edit/query budget 的动作；
- 与已查询的同一 sample/edit-state 完全重复的动作。

若 position mask 为空，只允许 STOP。Invalid action 不应被环境静默映射到其他 token。

## 12. Environment transition 与查询账本

一次 CONTINUE transition：

1. 应用 `(position, replacement)` 到 current tokens。
2. 用冻结 codec 解码。
3. 运行公共 RF projection。
4. 计算 RF metrics。
5. 查询 Victim A logits；计算新 margin 和 reward。

查询口径与 P6 对齐：

- clean codec baseline query 作为 setup query 单独报告，不计 candidate-search budget。
- 每个新的投影后 candidate state 计 1 个 logical Victim query。
- 完全相同的 sample/edit-state 可缓存；同时报告 raw requests、unique model calls 和 cache hits。
- STOP 不查询 Victim。
- 最终离线 B/C 审计查询不计攻击预算，但必须单独报告。
- 推理时最多 8 个编辑，因此 PPO 在线 Victim query 上限为 8，而训练总查询成本必须另行累计并报告。

## 13. Reward

定义 adversarial hardness：

\[
H_t=\max_{k\ne y}z_k-z_y,
\qquad
\Delta H_t=H_t-H_{t-1}.
\]

在 `generator_search_dev` 上预先冻结 robust scale (s_H\)，使用：

\[
\widetilde{\Delta H_t}=\operatorname{clip}(\Delta H_t/s_H,-5,5).
\]

草案 reward：

\[
r_t=
\lambda_H\widetilde{\Delta H_t}
-\lambda_Q c_t^{\mathrm{query}}
-\lambda_E c_t^{\mathrm{edit}}
-\lambda_{RF}V_t^{\mathrm{RF}}
-\beta_{KL}D_{KL,t}^{\mathrm{rep}}
+b_{\mathrm{success}}I_t^{\mathrm{success}}.
\]

其中训练 success 只定义为：Victim A misclassified AND digitally RF-valid AND budgets respected。它不包含 Evaluator B/C。RF penalty 只能依赖公共数字域指标；projection 后仍 invalid 时给予 penalty 并终止。

初始系数是待专家批准的 proposed values，而非已冻结事实：

- λH=1.0；λQ=0.01；λE=0.005；λRF=1.0；βKL=0.02；success bonus=1.0。
- 只允许在 `generator_search_dev` 进行一次小型预注册 grid/sensitivity；不得使用 Policy Gate 或 future PPO holdout 调参。
- 必须报告每一 reward component 的均值、标准差、p95 和占总 reward 比例，以检查 reward hacking。

## 14. PPO objective 与网络

Actor 使用三个 head 的联合 log-probability；critic 输出 (V_\psi(s_t)\)。PPO objective：

\[
L=L_{\mathrm{clip}}-c_VL_{\mathrm{value}}
+c_{H,s}\mathcal H_{\mathrm{stop}}
+c_{H,p}\mathcal H_{\mathrm{pos}}
+c_{H,r}\mathcal H_{\mathrm{rep}}
-\beta_{KL}D_{KL}^{\mathrm{rep}}.
\]

建议结构：

- frozen infill/reference feature extractor；
- trainable global state MLP；
- shared candidate-table attention/MLP；
- Stop Bernoulli head；
- masked Position categorical head；
- selected-position-conditioned Replacement categorical head；
- separate scalar critic head。

reference assets 全部冻结。policy residual output layers 零初始化。Victim、codec、projection 不进入 autograd graph。

## 15. 数据协议

- PPO training：`generator_train`，仅在专家确认其与 Victim/tokenizer 历史使用边界后启用。
- 超参数/early stopping：`generator_search_dev`；它是历史开发数据，不能声称独立验证。
- 当前 `policy_gate`：只保留 P5b/P6 历史证据，不用于 PPO reward tuning、checkpoint selection 或新的独立结论。
- PPO confirmatory holdout：**尚未分配，必须由专家决定并冻结**。建议从未用于攻击方法开发的来源建立新的 `ppo_gate`；若从 `defense_train_core` 划出，必须永久记录并从后续 defender-training pool 中剔除。
- `final_test`：继续封存，只在模型、阈值、统计和报告模板全部冻结后访问一次。

没有新的独立 PPO holdout 时，只允许把结果标为 development result，不允许宣称 PPO 已通过正式 Gate。

## 16. 训练与选择协议

获批后分三步，任何一步失败即停止：

1. **P7a interface smoke**：合成/开发样本验证 mask、joint log-prob、query ledger、KL only-on-replacement、无 B 访问。不得报告攻击结论。
2. **P7b short development run**：小规模 rollout，检查 reward component、policy entropy、clip fraction、approx KL、value loss、invalid action、RF violation 和 collapse。
3. **P7c frozen formal run**：所有超参数和 checkpoint-selection rule 锁定后，才在新 PPO holdout 与共同 baselines 上运行一次。

checkpoint selection 只能使用 development Victim/RF objective，不得读取 Evaluator B/C。checkpoint 冻结后才运行一次离线 identity audit。

## 17. 对照组与公平性

至少比较：

- query-matched random plausible；
- P6 Greedy；
- beam-coordinate（诊断性）；
- PPO three-head policy。

所有方法固定：source IDs、edit budget、candidate graph、candidate filters、RF projection、RF constraints 和统计脚本。报告：

- conditional valid-hard pooled/macro/per-device；
- paired bootstrap 95% CI；
- Victim margin reduction；
- B/C offline retention；
- RF-valid fraction 与完整 waveform metrics；
- inference logical/unique queries；
- PPO total training queries与 wall-clock；
- edit/position/replacement entropy；
- stop step distribution；
- reward/KL/clip/value diagnostics。

## 18. Proposed PPO Gate（待专家批准）

安全门：Evaluator B pooled/macro retention ≥98%，minimum-device retention ≥90%（沿用当前 P4 小样本阈值），RF-valid fraction ≥95%。P8/final 前仍必须增加 Evaluator C 或外部域复验。

在安全门通过后，满足以下至少一项：

1. **效果优势**：相同 edit/query cap 下，PPO 比最佳非 RL 搜索高 ≥5 pp，paired 95% CI 下界 >0；或
2. **摊销优势**：PPO 相对最佳非 RL 搜索 valid-hard non-inferiority margin 不低于 -2 pp，同时平均在线 Victim queries ≤其 25%，且报告总训练查询成本。

5 pp、-2 pp 和 25% 都是 proposed project thresholds，必须由专家明确批准或修改后才能冻结。

## 19. 必须生成的审计产物

若获批实施，每次 run 必须输出：

- 完整 config 与 SHA-256；
- data role IDs/hash；
- frozen codec/infill/graph/Victim hashes；
- source count、seed、edit/query budget；
- raw/unique/cache-hit query ledger；
- actor/critic/reward diagnostics；
- pooled/macro/per-device 指标与 paired CI；
- B/C access log，证明其未进入训练闭环；
- `final_test_used=false`；
- Gate status、reason、next_allowed_stage、prohibited_actions。

## 20. 实施前专家必须确认的未决项

1. 新 PPO confirmatory holdout 的来源与规模。
2. Position reference 新增 entropy/candidate-uncertainty 权重与 temperature。
3. reward 系数和唯一允许的 development sensitivity grid。
4. proposed PPO Gate 的 5 pp、-2 pp、25% 阈值。
5. P7-v1 是否保持 reference-frozen candidate set。
6. Evaluator C/外部域复验安排。

只有上述项目在评审清单中获得明确批准，才允许创建 PPO 实现分支。当前默认状态仍为 **NO-GO for implementation/training**。

