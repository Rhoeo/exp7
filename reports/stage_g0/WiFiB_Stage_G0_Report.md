# WiFi-B Stage G0 baseline report

## Frozen setup

- Dataset: 17 local WiFi-B device files, 59,409 frames.
- Input: 2,048 complex samples starting at complex sample 128.
- Normalization: per-frame `mean(I² + Q²) = 1`.
- Model selection: reward-validation chronological block only.
- Final test: later chronological block, never used for checkpoint selection.

## Results

| Model | Role | Parameters | Reward validation | Defense split | Final test |
|---|---|---:|---:|---:|---:|
| Victim A | attack-visible time-domain CNN | 78,257 | 90.54% | 89.53% | 86.70% |
| Evaluator B | hidden FFT multi-scale CNN | 170,237 | 99.73% | 99.64% | 98.59% |

Macro-F1 on final test is 85.85% for Victim A and 98.58% for Evaluator B.

## Interpretation and gates

Victim A is adequate for the first attack baseline, but it is not uniformly
strong across all devices. Its weakest final-test recalls are Device 41
(35.52%), Device 30 (58.50%), and Device 6 (62.61%). Therefore attack success
rate must be computed on samples that Victim A classifies correctly before the
attack. Overall all-sample accuracy must be reported separately.

Evaluator B is substantially stronger and architecturally independent: it uses
FFT real/imaginary components and standardized log magnitude rather than the
Victim A time-domain stream. It stays hidden from the attack optimizer and is
used to reject victim-specific adversarial artifacts.

The 98.59% Evaluator B final accuracy is still a same-day result. It may exploit
stable acquisition or channel characteristics. Cross-day and cross-receiver
claims must use the ManyTx partitions; the WiFi-B result alone cannot establish
receiver-agnostic RF fingerprinting.

## Frozen artifacts

- `runs/stage_g0/frozen/wifib_v1/victim_a.pt`
- `runs/stage_g0/frozen/wifib_v1/evaluator_b.pt`
- `runs/stage_g0/frozen/wifib_v1/frozen_models.json`

These files are immutable inputs to the FGSM/PGD baseline, VQ-VAE identity
gate, and later PPO reward experiments. Any retrained model must receive a new
version rather than replacing `wifib_v1`.
