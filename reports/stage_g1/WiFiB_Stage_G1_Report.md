# WiFi-B Stage G1：RF Tokenizer 可行性报告

## 结论

Stage G1 的核心假设已在完整 `reward_validation` 上通过：连续 IQ 可以被离散化为 RF token，并在解码后保持设备身份。冻结配置为 polyphase VQ-VAE，`patch_size=1`、`K=1024`、`latent_dim=2`，每个 2048 点波形对应 2048 个 token。

训练只使用波形、频谱、相关性、功率和 VQ commitment 损失，未向 tokenizer 提供 Victim A 或 Evaluator B 的梯度。这使身份 Gate 是独立检验，而不是分类器蒸馏结果。

## 决策过程

| 方案 | 重构 SNR | Victim A 下降 | Evaluator B 下降 | 码本状态 | Gate |
|---|---:|---:|---:|---|---|
| 四级卷积 AE，转置卷积解码 | 17.60 dB | 51.07 pp | 32.71 pp | 不适用 | FAIL |
| 四级卷积 AE，sub-pixel 解码 | 18.16 dB | 51.95 pp | 35.06 pp | 不适用 | FAIL |
| 可逆 polyphase AE，patch=4 | 120 dB（逐点一致） | 0.00 pp | 0.00 pp | 不适用 | PASS |
| polyphase VQ-VAE，patch=4，K=256 | 16.57 dB | 73.73 pp | 27.44 pp | 254/256 活跃 | FAIL |
| polyphase VQ-VAE，patch=1，K=1024 | 30.36 dB | 1.63 pp | 0.00 pp | 1024/1024 活跃 | **PASS** |

失败实验说明：约 18 dB 的重构质量不足以保留 Victim A 所利用的弱指纹；patch=4 的失败并非码本塌缩，而是单码容量不足。可逆 packing 把连续编码损失和离散量化损失分离后，patch=1/K=1024 达到了足够的量化精度。

## 完整 reward-validation 指标

- 重构 SNR：30.36 dB
- 波形相关系数：0.9995
- 频谱 log-magnitude L1：0.01267
- Victim A：0.9054 → 0.8892，下降 1.63 个百分点
- Victim A 干净正确样本身份保持率：0.9689
- Evaluator B：0.9973 → 0.9973，下降 0.00 个百分点
- Evaluator B 干净正确样本身份保持率：0.9997
- 码本：1024/1024 活跃，perplexity 981.66，最大单码占比约 0.003

## 冻结与下一阶段约束

冻结检查点为 `runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt`，SHA-256 为 `331ff5136efd8ed45ef444b348e7bb51b0b77a8e1da5523a4fbeb40ae0e3607e`。

进入 RF-GPT 后 tokenizer 必须冻结。基础 RF-GPT 只在 `generator_train` token 上做 device-conditioned next-token 建模；`reward_validation` 只作 checkpoint 选择；`final_test` 在最终锁模前不使用。Evaluator B 始终对生成器和后续 PPO 隐藏。

当前代价是序列长度 2048。它适合先验证 conditional generation，但全量训练最好使用更强 GPU。后续若压缩上下文，优先尝试 product/residual VQ，不再回到已证伪的单码 patch=4 配置。

