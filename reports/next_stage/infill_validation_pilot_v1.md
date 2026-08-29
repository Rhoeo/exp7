# P3 Masked-Infill Smoke Validation

- Gate: **SMOKE_PASS**
- Validation examples / masked targets: 512 / 1185
- Loss: 6.9274
- Original-token top-1 / top-5: 0.003 / 0.008
- Mean transition-supported latent candidates before probability filter: 12.03
- Provisional smoke delta_ll: `1.0`
- Coverage/no-op at provisional threshold: 0.991 / 0.009

No Victim A, Evaluator B, policy-Gate attack outcome, or final-test feedback was used. This is a CPU smoke result, not the formal Infill Gate; the relative-probability threshold remains provisional until P4 waveform checks are available.
