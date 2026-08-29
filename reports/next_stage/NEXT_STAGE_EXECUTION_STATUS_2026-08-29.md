# AdvTG 同构射频指纹对抗样本方案：执行状态（2026-08-29）

## 当前结论

P0–P6 已完成。P6 在冻结 untouched-buffer Policy Gate 上通过项目 Gate Q：预注册 Greedy 查询搜索在 Q=256 时相对同预算 query-matched random 提升 8.25 个百分点，paired bootstrap 95% CI 为 [5.83, 10.92] pp；同时 Q=256 random 的 6.55% 成功率，Greedy 在 Q=128 用一半查询达到 8.98%。这证明 Victim A margin 对当前候选动作存在可利用的黑盒信号。

Gate Q PASS 不等于 PPO 授权。当前仍停在“新 PPO 三头规格提交专家评审”边界，未实现、未训练 PPO，未读取 `final_test`。

## 阶段状态

| 阶段 | 状态 | 主要证据 |
|---|---|---|
| P0 冻结 | PASS | `current_state_manifest.json`、`current_state_summary.md`；初始测试通过 |
| P1 数据角色 | PASS with limitation | untouched buffer 1,779 条；Device 11 合格样本不足，Policy Gate 固定为 25/device=425 |
| P2 候选图 | PASS | 1024×32 图；transition-supported top-16=0.741；无孤立码字 |
| P3 masked infill | PASS（hybrid v2） | generator_search_dev 上 top-1=18.82%、top-5=53.50%；ΔLL=4 已冻结 |
| P4 RF 流水线 | PASS | 公共 decode→projection→metric→classifier 顺序；数字域 waveform-valid |
| P5a/P5b | PASS | P5b 固定 Gate、3 seeds、412 codec-joint denominator；B/RF 保留率满足阈值 |
| P6 查询 Gate | PASS | Greedy Q1/Q2 PASS；beam-coordinate 未超过 random |
| P7 PPO | 规范草案待审；禁止实施 | 三头 PPO spec、配置草案和评审清单已提交，需专家逐项批准 |
| P8/final_test | 未执行 | 生成器/搜索器及评估协议冻结后再决定 |

## P6 关键表（conditional valid-hard rate）

| 查询预算 | query-matched random | Greedy | beam-coordinate |
|---:|---:|---:|---:|
| 64 | 5.18% | 5.83% | 1.21% |
| 128 | 6.15% | 8.98% | 1.70% |
| 256 | 6.55% | 14.81% | 2.91% |

所有 P6 方法的 Evaluator B retention 与 RF-valid fraction 均为 100%；B 只做最终离线身份审计，不进入候选、搜索、奖励或在线过滤。

## 已知困难与解释

1. Policy Gate 只能平衡到 25/device，原因是 Device 11 在 untouched buffer 中只有 25 条双模型 clean-correct 样本；没有有放回补样，也没有回退到历史 validation。
2. masked Transformer 单独训练的数据效率不足，已改为冻结 bidirectional transition prior + Transformer residual 的 hybrid v2；这不是自由生成器，仍只做 reference-conditioned token infill。
3. beam-coordinate 明显弱于随机完整 8-edit 状态，说明当前动作空间可能具有非局部组合性；因此不能把“Greedy 有信号”外推成“RL 一定有收益”。
4. 当前 RF 结论仅是数字域 waveform-valid，尚未包含 DAC/PA/信道/空口验证；Evaluator B 也不是最终身份真值。
5. P6 v0 曾因 beam 随预算只实现 2.85/4.54/6.84 个编辑而违反公平编辑预算，已保存为无效记录；v2 固定 cardinality 后才纳入正式结果。

## 复现实验与报告

- [P6 正式 JSON 报告](E:/exp7/reports/next_stage/p6_query_gate_report.json)
- [P6 正式 Markdown 报告](E:/exp7/reports/next_stage/p6_query_gate_report.md)
- [P6 诊断报告](E:/exp7/reports/next_stage/P6_DIAGNOSTIC_REPORT_2026-08-29.md)
- [P6 v2 配置](E:/exp7/configs/attack/wifib_p6_query_gate_v2.json)
- [P6 无效 v0 记录](E:/exp7/reports/next_stage/p6_query_gate_report_beam_growth_invalid_v0.json)
- [P7 三头 PPO 规范草案](E:/exp7/reports/next_stage/P7_THREE_HEAD_PPO_SPEC_DRAFT_2026-08-29.md)
- [P7 专家评审清单](E:/exp7/reports/next_stage/P7_PPO_EXPERT_REVIEW_CHECKLIST_2026-08-29.md)
- [P7 配置草案](E:/exp7/configs/ppo/wifib_three_head_ppo_spec_draft_v1.json)
- [P5b formal 报告](E:/exp7/reports/next_stage/p5b_formal_report.md)
- [完整执行计划](E:/exp7/reports/next_stage/NEXT_STAGE_IMPLEMENTATION_PLAN.md)

全量单元测试：36 项通过。P6 总共记录 523,799 个跨方法/seed 去重后的投影候选状态；`final_test` 信号数据访问为 false。
