# P4 Waveform Validity and Infill Gate

- Gate: **FAIL**
- Development sources: 170 (10/device)
- Victim A queries: **0**
- Policy Gate used: **false**
- Final test used: **false**
- Candidate coverage after latent/transition + probability + actual-decoding checks: 0.816
- Cross-entropy gain over uniform: 3.4189 nats
- Max-edit B pooled/macro/min-device retention: 1.000 / 1.000 / 1.000
- Infill peak/derivative p95 improvement over full-codebook random diagnostic: 0.924 / 0.924
- Failed checks: `candidate_position_coverage`

This Gate evaluates digital waveform plausibility only. Evaluator B is a preliminary offline auditor and no over-the-air feasibility is claimed.
