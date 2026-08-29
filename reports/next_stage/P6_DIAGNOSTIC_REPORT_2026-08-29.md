# P6 查询搜索诊断报告（2026-08-29）

## 结论

P6 v2 在冻结的 untouched-buffer Policy Gate 上完成。使用同一候选池、8-token 编辑上限、公共 RF 投影和每源 Victim A 查询预算时，预注册 Greedy 搜索满足 Gate Q1 和 Gate Q2；beam-coordinate 未超过 query-matched random。Gate Q 的 PASS 只表示“黑盒查询分数具有可利用信号”，不授权实施或训练 PPO。

正式分母为 412 条 codec-joint-correct source，指标是 **conditional valid-hard rate on dual-clean-correct sources**。`final_test` 未读取。

## 公平比较

| Q | query-matched random | Greedy | Greedy−random (pp) | paired 95% CI (pp) | beam-coordinate | beam−random (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5.18% | 5.83% | +0.65 | [-0.57, 1.94] | 1.21% | -3.96 |
| 128 | 6.15% | 8.98% | +2.83 | [1.21, 4.61] | 1.70% | -4.45 |
| 256 | 6.55% | 14.81% | **+8.25** | **[5.83, 10.92]** | 2.91% | -3.64 |

Gate Q1：Greedy 在 Q=256 比 query-matched random 高 8.25 pp，paired CI 下界为 5.83 pp，PASS。

Gate Q2：query-matched random Q=256 的 valid-hard 为 6.55%；Greedy Q=128 已达到 8.98%，查询预算为其 50%，PASS。

Greedy Q=256 的平均逻辑查询为 208.5/source，beam Q=256 为 256/source；所有报告方法的 Evaluator B retention 和 RF-valid fraction 均为 100%。

## 诊断含义

1. P5b 的零查询 random plausible 只能作为辅助对照，不能直接用于 Gate Q。加入同预算的随机候选状态后，random baseline 在 Q=256 达到 6.55%，因此本报告没有把较弱的零查询基线当成 PPO 依据。
2. Greedy 的效果随查询预算明显上升（5.83% → 8.98% → 14.81%），说明 Victim A margin 对候选动作存在可利用的黑盒信号；但结果仍是条件成功率，不外推为全分布攻击率。
3. beam-coordinate 低于 random（1.21%/1.70%/2.91%）。这更像是离散动作空间的非局部性：随机完整 8-edit 状态能跳到组合区域，而单坐标邻接 beam 在固定预算内难以到达；不能据此宣称 token 空间已经适合 RL。
4. P6 v2 的 beam 状态固定为 8 个编辑；clean stop 仍被允许，所以 beam 的实际平均编辑数低于 8。Greedy/random 使用同一最大编辑预算，且所有结果都按公共投影后的 waveform-valid 与 B 审计计算。

## 协议修正记录

P6 v0 曾按逐层增加编辑实现 beam，导致 Q=64/128/256 平均只实现约 2.85/4.54/6.84 个编辑，和固定 8-edit random 不公平。该运行已保存为 `p6_query_gate_report_beam_growth_invalid_v0.*`，不纳入结论。v2 改为固定 cardinality 后才作为正式结果。

## 下一步边界

- P0–P6 结果与配置、数据 split hash、checkpoint hash 和单元测试均已冻结。
- Gate Q PASS 后，只能进入新的 PPO 三头规格的专家评审边界；当前不实现、不训练 PPO，也不使用 Evaluator B 进入状态、奖励或在线过滤。
- 若专家不批准 PPO spec，应回到 P2–P6 诊断（候选覆盖、动作组合非局部性、Greedy 的位置/候选集中度和跨设备稳定性）。
- P8 防御效用和 `final_test` 仍保持条件执行与封存状态。

