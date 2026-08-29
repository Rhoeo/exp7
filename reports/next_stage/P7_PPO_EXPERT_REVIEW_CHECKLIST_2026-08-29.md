# P7 三头 PPO 专家评审清单

状态：**NOT APPROVED**  
对应规范：`P7_THREE_HEAD_PPO_SPEC_DRAFT_2026-08-29.md`  
对应配置：`configs/ppo/wifib_three_head_ppo_spec_draft_v1.json`

请专家对每项选择 `APPROVE`、`REVISE` 或 `REJECT`，并给出必要修改。只有所有 blocking 项被明确批准，才允许创建 PPO 实现分支。

## A. 研究与数据边界（blocking）

- [ ] A1. 同意 P7 只研究 score-based black-box untargeted token editing。
- [ ] A2. 同意 `generator_train` 作为拟议 PPO training role；或指定替代来源。
- [ ] A3. 指定新的 PPO confirmatory holdout 来源、每设备数量和冻结规则。
- [ ] A4. 同意当前 P5b/P6 Policy Gate 只保留为历史证据，不用于 PPO tuning/checkpoint selection。
- [ ] A5. 同意 `final_test` 继续封存。

专家意见：

```text

```

## B. 三头动作定义（blocking）

- [ ] B1. Stop 为独立 Bernoulli head。
- [ ] B2. Position 为最多 64 个位置上的 masked categorical head。
- [ ] B3. Replacement 为所选位置最多 4 个 plausible candidates 上的 masked categorical head。
- [ ] B4. PPO ratio 使用 Stop/Position/Replacement 的完整联合 log-probability。
- [ ] B5. scalar value head 是 critic，不属于“三个 actor head”。
- [ ] B6. 同一位置最多编辑一次，最多编辑 8 个位置。

专家意见：

```text

```

## C. Reference policy 与 KL（blocking）

- [ ] C1. Position reference prior 使用 infill entropy/candidate uncertainty 加 P6 无 Victim 排序特征。
- [ ] C2. 批准或修改 Position prior 的新增权重和 temperature。
- [ ] C3. Replacement reference 只在 graph∩infill∩RF-precheck 候选上归一化。
- [ ] C4. 只对 Replacement head 加 reference KL。
- [ ] C5. Position/Replacement residual head 零初始化。
- [ ] C6. Stop head 不使用伪造 reference distribution，也不加 reference KL。

专家意见：

```text

```

## D. 状态、奖励与查询（blocking）

- [ ] D1. 状态不包含 Victim gradients/hidden embeddings。
- [ ] D2. Evaluator B/C 完全排除在 observation、reward、mask、early stop 和 model selection 外。
- [ ] D3. 每次 sampled edit 在 projection 后只查询一次 Victim A；STOP 为零查询。
- [ ] D4. clean setup query、attack queries、B/C audit queries 和 total training queries 分开报告。
- [ ] D5. 批准或修改 reward component 与 proposed coefficient。
- [ ] D6. 训练 success 只由 Victim A、RF-valid 和 budget 决定。

专家意见：

```text

```

## E. 候选与公平对照（blocking）

- [ ] E1. P7-v1 使用 episode 内 reference-frozen candidate set。
- [ ] E2. 若启用 dynamic local re-infill，同意同时重跑 random/Greedy/PPO。
- [ ] E3. 同意所有方法共享 source IDs、edit budget、candidate filters、projection 和统计脚本。
- [ ] E4. 同意 query-matched random、P6 Greedy、beam diagnostic 为最低对照集合。

专家意见：

```text

```

## F. PPO Gate（blocking）

- [ ] F1. 批准或修改 safety gate：B pooled/macro ≥98%、minimum-device ≥90%、RF-valid ≥95%。
- [ ] F2. 批准或修改效果优势门：≥5 pp 且 paired CI 下界 >0。
- [ ] F3. 批准或修改摊销门：non-inferiority ≥-2 pp 且 online queries ≤最佳非 RL 的25%。
- [ ] F4. 确认 P8/final 前需要 Evaluator C 或外部域复验。

专家意见：

```text

```

## 最终决定

- [ ] **APPROVE FOR P7a INTERFACE SMOKE ONLY**
- [ ] **REVISE AND RESUBMIT**
- [ ] **REJECT / RETURN TO P2–P6 DIAGNOSIS**

批准人：  
日期：  
批准范围与附加限制：

```text

```

在最终决定被填写前，`implementation_authorized=false`、`training_authorized=false` 保持不变。

