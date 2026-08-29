# P3 Masked-Infill Smoke Validation

- Gate: **SMOKE_PASS**
- Validation examples / masked targets: 512 / 1185
- Loss: 3.5125
- Original-token top-1 / top-5: 0.188 / 0.535
- Mean transition-supported latent candidates before probability filter: 12.03
- Provisional smoke delta_ll: `4.0`
- Coverage/no-op at provisional threshold: 0.976 / 0.024

No Victim A, Evaluator B, policy-Gate attack outcome, or final-test feedback was used. This is a CPU smoke result, not the formal Infill Gate; the relative-probability threshold remains provisional until P4 waveform checks are available.
