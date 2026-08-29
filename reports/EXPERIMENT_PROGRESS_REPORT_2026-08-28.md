# 射频指纹对抗样本项目实验进展报告

日期：2026-08-28  
状态：按要求暂停，等待专家讨论  
工作区：`E:\exp7`

## 1. 执行摘要

当前已经完成了从数据缓存、冻结分类器、VQ tokenizer 到 RF-GPT pilot 和攻击基线的第一轮闭环。最重要的结论有三条：

1. **VQ tokenizer 可行。** 可逆 polyphase VQ-VAE（patch=1、K=1024）在完整 `reward_validation` 上达到 30.36 dB 重构 SNR；Victim A 准确率下降 1.63 个百分点，Evaluator B 下降 0，身份保持率分别为 96.89% 和 99.97%。该 tokenizer 已冻结。
2. **RF-GPT 可以学到局部 RF token 语法和设备条件，但自由生成失败。** 4k pilot 的 teacher-forced 验证 perplexity 为 10.83；正确设备条件相对打乱条件的 NLL 优势为 0.26455 nat，说明标签确实被使用。但从单 token 或 256-token 前缀自由滚动生成 2048 token 时，A/B 条件身份都接近失败。
3. **参考 token 编辑有可用动作，但尚未优于简单基线。** RF-GPT 2%–5% 稀疏编辑在 22–28 dB 失真下能产生 5.4%–8.9% 的双重有效困难样本（Victim 被翻转、Evaluator 保持身份，且扰动经过 22 MHz 带限）。然而均匀随机 token 替换在匹配 SNR 下达到 5.4%–12.5%，Victim-visible PGD 在相同样本上达到 100%。因此现在**不能宣称 RF-GPT 已优于 baseline，也没有开始 PPO**。

## 2. 数据与缓存

本地数据已从 `E:\data` 迁移到 `D:\data`，当前配置为 `configs/data/rffi_data_v1.json`。

| 数据 | 状态 |
|---|---|
| WiFi-B | 17 个设备、59,409 帧、每帧原始复 IQ 长度 17,550；使用 start=128、长度=2,048 的窗口 |
| ManyTx | 已转换为连续 memmap；IQ cache 形状 `(1,020,643,256, 2)`，约 1.95 GiB |
| WiFi-B IQ window cache | 形状 `(59,409, 2, 2,048)`，约 0.91 GiB |
| WiFi-B RF token cache | `E:\data_cache\rffi_v1\tokens\wifib_vq_p1_k1024\tokens.npy`，形状 `(59,409, 2,048)`、`uint16`，约 232 MiB |

WiFi-B split 行数：`generator_train=31,486`、`reward_validation=8,914`、`defense_train=8,913`、`final_test=8,317`，另有三段 buffer。所有模型选择均未使用 `final_test`。

数据限制：WiFi-B 是单日数据，当前 final-test 是同一天的时间后置块，不等价于跨天泛化；Evaluator B 的同日准确率很高，可能含采集域特征。ManyTx 尚未进入本轮生成器实验。

## 3. Stage G0：冻结分类器

冻结模型位于 `runs/stage_g0/frozen/wifib_v1/`。

- Victim A：时域 residual CNN；reward-validation accuracy 0.9054，final-test accuracy 0.8670。
- Evaluator B：FFT 多尺度 CNN；reward-validation accuracy 0.9973，final-test accuracy 0.9859。
- Victim A 对 Device41、Device30、Device6 的召回较弱，因此攻击成功率统一按“clean-correct source samples”统计。
- 本地 CUDA 缺少 `nvrtc-builtins64_121.dll`，复杂数 `abs` 使用手动 `sqrt(real²+imag²)` 绕过；训练和评估已验证通过。

## 4. Stage G1：tokenizer 结果

### 通过的配置

冻结检查点：`runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt`  
SHA-256：`331ff5136efd8ed45ef444b348e7bb51b0b77a8e1da5523a4fbeb40ae0e3607e`

完整 reward-validation（8,914 条）：

- 重构 SNR 30.36 dB，波形相关 0.9995，频谱 log-L1 0.01267；
- Victim A：0.9054 → 0.8892，下降 1.63 pp；clean-correct identity retention 0.9689；
- Evaluator B：0.9973 → 0.9973，下降 0.00 pp；clean-correct identity retention 0.9997；
- K=1024 全部活跃，perplexity 981.19（全量 token cache）。

### 失败配置及其含义

| 配置 | 重构 SNR | Victim A drop | Evaluator B drop | 结论 |
|---|---:|---:|---:|---|
| 四级卷积 AE + 转置卷积 | 17.60 dB | 51.07 pp | 32.71 pp | 结构性失真，FAIL |
| 四级卷积 AE + sub-pixel | 18.16 dB | 51.95 pp | 35.06 pp | 仍 FAIL |
| polyphase AE，patch=4 | 逐点恒等 | 0 | 0 | 证明数据流/评估链正确 |
| polyphase VQ，patch=4、K=256 | 16.57 dB | 73.73 pp | 27.44 pp | 码本不塌缩，但单码容量不足 |
| polyphase VQ，patch=1、K=1024 | 30.36 dB | 1.63 pp | 0 pp | **PASS** |

详细报告：[WiFiB_Stage_G1_Report.md](E:\exp7\reports\stage_g1\WiFiB_Stage_G1_Report.md)。

## 5. Stage G3：RF-GPT pilot

模型：96 维、3 层、4 heads、context=256、约 575k 参数；训练集 4,096、验证集 1,024，未使用 A/B 分类器反馈。验证 perplexity 从 smoke 的 707.7 降到 10.83，token accuracy 28.3%。

条件敏感性：正确条件 NLL 2.39418，打乱设备标签 NLL 2.65873，差值 0.26455 nat，判定为 strong condition usage。

### 自由生成结果

- temperature=0.7、top-p=0.9：Victim A 17 个样本中正确 3 个（17.6%），Evaluator B 正确 0 个；
- greedy：A/B 都为 11.8%；
- 256-token reference prefix 续写：Victim A 11.8%，Evaluator B 5.9%；
- 样本序列均唯一、无 generator-train 精确重复，但这不能弥补身份失败。

解释：模型对 teacher-forced 的局部 next-token 预测有效，但自由 rollout 发生误差累积并离开真实 RF 流形。不能把该 unconditional generator 直接用于 PPO。

## 6. Stage G4：参考 token 编辑与基线

样本：`reward_validation` 中每设备 4 条，共 68 条；只在 clean-correct 条件上统计攻击成功；Evaluator B 从未参与 proposal 或梯度优化。带限版本对扰动做 22 MHz 中心带投影，采样率 35 MHz。

### 22 MHz 带限后的结果

| 方法 | 编辑/预算 | 实际扰动 SNR | Victim fool | Evaluator 保持 | 双重有效率 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| RF-GPT 编辑 | 2%（41 token） | 28.34 dB | 5.4% | 98.5% | 5.4% | 扰动 PAPR≈431 |
| RF-GPT 编辑 | 3%（61 token） | 27.63 dB | 7.1% | 98.5% | 7.1% | 扰动 PAPR≈446 |
| RF-GPT 编辑 | 5%（102 token） | 24.54 dB | 8.9% | 95.6% | 8.9% | 扰动 PAPR≈330 |
| 均匀随机 token | 0.1%（2 token） | 28.28 dB | 5.4% | 100.0% | 5.4% | PAPR≈952 |
| 均匀随机 token | 0.2%（4 token） | 26.18 dB | 7.1% | 100.0% | 7.1% | PAPR≈621 |
| 均匀随机 token | 0.3%（6 token） | 24.37 dB | 12.5% | 98.5% | 12.5% | PAPR≈482 |
| Victim-visible PGD | 26 dB 预算 | 27.60 dB | 100.0% | 98.5% | 100.0% | PAPR≈26.7 |
| Victim-visible PGD | 22 dB 预算 | 25.20 dB | 100.0% | 97.0% | 100.0% | PAPR≈26.1 |

RF-GPT 的优点目前是编辑更分散、token proposal 不需要 Victim 梯度；但其扰动 PAPR 仍高，且攻击效率未超过随机 token。PGD 只是 white-box 上界，不应与 black-box RF-GPT 直接做唯一结论。

详细结果：[WiFiB_Stage_G4_Hardness_Report.md](E:\exp7\reports\stage_g4\WiFiB_Stage_G4_Hardness_Report.md)。

## 7. 遇到的主要困难

### 7.1 连续重构 SNR 与身份保持不等价

17–18 dB 的 AE 重构看起来数值不错，但 A/B 身份大幅下降；Victim A 依赖的硬件微弱特征对局部结构失真很敏感。最终使用可逆 polyphase packing，把连续编码损失与量化损失分开。

### 7.2 离散 token 的容量与序列长度冲突

patch=4、K=256 的单码容量不足；patch=1、K=1024 能通过 Gate，但序列长度为 2,048。后续若要压缩上下文，需要 product/residual VQ 或多码本 token，不能简单增大 patch。

### 7.3 RF-GPT teacher forcing 与自由 rollout 的落差

局部 next-token perplexity 已降到 10.83，且设备条件敏感，但自由生成仍漂移。单 token 起始分布不足以锁定整段 WiFi preamble；256-token prefix 也不够。对于对抗样本，更合适的形态可能是 reference-conditioned token edit/infill，而非从设备 ID 无条件生成整段波形。

### 7.4 生成模型没有自动战胜简单基线

同一 SNR 下，均匀随机 token 替换的 fool rate 不低于 RF-GPT；Victim-visible PGD 更强。只有在明确的带宽、峰值、PAPR、EVM、PA 非线性和协议约束下，才能检验 RF-GPT 是否有“低可检测性/低尖峰”的真实优势。

### 7.5 远程 AutoDL 实例当前不可用

已成功连接，并仅创建了 `/tmp/exp7`；但该实例没有 `/dev/nvidia0`，`nvidia-smi` 返回 “No devices were found”，默认环境没有 Python/Conda。因此没有上传数据或代码，也没有影响其他实验。若专家建议做 full training，需要先启动带 GPU/Python 的实例。

## 8. 暂停时的代码与产物

- G0：`rffi_core/victim/`、`runs/stage_g0/frozen/wifib_v1/`
- G1 tokenizer：`rffi_core/generators/vqvae/`
- token cache：`rffi_core/generators/vqvae/cache_tokens.py`、`rffi_core/data/token_datasets.py`
- RF-GPT：`rffi_core/generators/rfgpt/`
- token edit probe：`rffi_core/generators/rfgpt/probe_token_edits.py`
- PGD baseline：`rffi_core/attacks/pgd_baseline.py`
- RF constraints：`rffi_core/attacks/rf_constraints.py`
- 测试：17 项通过；最近一次 `python -m unittest discover -s tests -v` 全部 OK。

## 9. 建议和专家讨论的问题

1. **威胁模型**：最终论文是 white-box、black-box 还是 transfer attack？若是 black-box，需要固定 Victim query budget；PGD 只能作为 white-box upper bound。
2. **RF 约束**：应采用哪些可接受范围——22 MHz 频谱 mask、最大瞬时幅度、扰动 PAPR、EVM、ACLR、PA/ADC 非线性、协议字段一致性？当前带限只是第一层约束。
3. **生成形态**：是否接受 reference-conditioned token edit/infill 作为 AdvTG 同构方案？若必须 unconditional device-conditioned generation，当前 G3 结果应判 No-Go。
4. **token 结构**：是否允许 2,048 token 的长上下文？若不允许，应优先研究 product/residual VQ，并用完整身份 Gate 复核。
5. **数据泛化**：是否需要把 ManyTx 纳入 tokenizer/GPT，或补充跨天/跨接收机数据后再做最终结论？
6. **PPO Go/No-Go**：只有当 reference-policy PPO 在相同 B 保持率、RF 约束和 query budget 下超过均匀 token 搜索，才值得投入 PPO；否则应保留简单搜索/PGD 作为主结果。

## 10. 建议暂停点

当前最稳妥的状态是：冻结 G1 tokenizer，保留 G3/G4 pilot 与 PGD/随机基线，**暂停 full RF-GPT、PPO 和 final-test 评估**，先由专家确认威胁模型与 RF 约束。专家确认后再按 Go/No-Go 选择：

- black-box reference edit → 固定 query budget，做 PPO 或 bandit policy；
- unconditional generation → 先解决 rollout/跨域身份失败；
- white-box 目标 → 直接以带限 PGD/UAP 为主基线，RF-GPT 作为可选 transfer 方法。

