# WiFi-B Stage G3/G4：RF-GPT 生成与编辑可行性

## 结论

当前小型 RF-GPT 学会了局部 RF token 语法和 device condition，但不能从单个起始 token 自由滚动生成身份正确的 2048-token 波形。参考前缀续写也未解决序列漂移。因此不把 unconditional RF-GPT 标记为成功，也不直接对它做 PPO。

把 RF-GPT 用作“参考波形的稀疏 token 编辑器”后，动作空间出现了可用的困难样本：在 68 条平衡的 `reward_validation` 样本上，2%/3%/5% 编辑的 Evaluator B 身份保持为 100%/98.5%/97.1%，Victim A fool rate 为 7.1%/12.5%/16.1%，双重有效困难率为 7.1%/12.5%/14.3%。

但同一位置数量级的均匀随机 token 替换在匹配失真时也能达到相近或更高 fool rate，而且 Victim-visible PGD 更强。因此 RF-GPT 当前的证据是“提供低尖峰、较高 RF 相似性的黑盒动作提议”，不是“已经优于攻击基线”。

## 对照结果

| 方法 / 预算 | 实际扰动 SNR | Victim A fool | Evaluator B 保持 | 双重有效率 | 备注 |
|---|---:|---:|---:|---:|---|
| RF-GPT 编辑 2%（41 token） | 26.65 dB | 7.1% | 100.0% | 7.1% | 扰动峰值均值 0.99 |
| RF-GPT 编辑 3%（61 token） | 25.62 dB | 12.5% | 98.5% | 12.5% | 扰动峰值均值 1.16 |
| RF-GPT 编辑 5%（102 token） | 22.66 dB | 16.1% | 97.1% | 14.3% | 扰动峰值均值 1.34 |
| 均匀随机 0.1%（2 token） | 26.26 dB | 7.1% | 100.0% | 7.1% | 扰动峰值均值 1.75，PAPR 1515 |
| 均匀随机 0.2%（4 token） | 24.17 dB | 14.3% | 100.0% | 14.3% | 扰动峰值均值 1.88，PAPR 988 |
| 均匀随机 0.3%（6 token） | 22.35 dB | 21.4% | 97.1% | 19.6% | 扰动峰值均值 2.05，PAPR 768 |
| Victim-visible PGD，26 dB 预算 | 27.20 dB | 100.0% | 100.0% | 100.0% | 峰值均值 0.19，PAPR 26.7 |
| Victim-visible PGD，22 dB 预算 | 24.37 dB | 100.0% | 98.5% | 100.0% | 峰值均值 0.26，PAPR 23.6 |

所有结果均在 68 条平衡样本上，攻击成功率只对干净时 Victim 正确的样本统计；Evaluator B 未参与 proposal 或梯度优化，仅做审计。PGD 的“dual valid”分母为 Victim A 和 Evaluator B 同时干净正确的样本。

## 对 RF-GPT 的 Go / No-Go

### Go：保留为黑盒/梯度不可用的候选 policy

- 编辑动作不需要 Victim A 梯度；
- 生成误差比均匀 token 替换更分散，峰值与扰动 PAPR 明显更低；
- 在 2%–5% 编辑率存在身份保持与 Victim fool 的交集，可作为 PPO 的初始探索分布。

### No-Go：当前条件下不宣称生成模型优于基线

- 自由生成的 conditional identity 失败；
- 同失真均匀 token baseline 的 fool rate 不低于 RF-GPT；
- 白盒 PGD 明显更强，且峰值扰动并不比 RF-GPT 更差。

## 下一步

在决定 PPO 之前，先固定 RF 约束（扰动带宽、峰值、PAPR、功率和协议窗口）并增加一个黑盒查询预算基线。PPO 的成功标准应至少是：在相同 Evaluator 保持率、相同 RF 预算和相同 Victim 查询次数下，双重有效困难率显著超过均匀 token 搜索；否则回退到直接优化/随机搜索等更简单方案。所有调参继续只用 `reward_validation`，最终结论才在锁模后查看 `final_test`。

