# P6 Query Search Gate

- Gate Q: **FAIL**
- Gate Q1/Q2: **FAIL / FAIL**
- Sources: 425 (25/device), fixed untouched-buffer policy Gate
- Codec-joint-correct denominator: 412
- Edit budget: 8 tokens
- Final test used: **false**
- Evaluator B: offline final audit only

| method | Q cap | valid-hard | 95% CI | mean queries | B retention | min-device B | RF valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| query-matched random | 64 | 0.052 | [0.032, 0.073] | 64.0 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 64 | 0.022 | [0.010, 0.036] | 64.0 | 1.000 | 1.000 | 1.000 |
| query-matched random | 128 | 0.061 | [0.040, 0.084] | 128.0 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 128 | 0.034 | [0.017, 0.051] | 128.0 | 1.000 | 1.000 | 1.000 |
| query-matched random | 256 | 0.066 | [0.043, 0.089] | 256.0 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 256 | 0.036 | [0.019, 0.053] | 224.0 | 1.000 | 1.000 | 1.000 |

| Q | beam - random (pp) | paired 95% CI (pp) | Gate Q1 |
|---:|---:|---:|---:|
| 64 | -2.99 | [-4.61, -1.54] | FAIL |
| 128 | -2.75 | [-4.45, -1.29] | FAIL |
| 256 | -2.91 | [-4.69, -1.21] | FAIL |

## Interpretation boundary

The random baseline receives the same candidate pool, edit budget, projection, and per-source query cap. P5b zero-query random results are auxiliary and do not decide Gate Q. Candidate proposals are frozen from the clean reference during P6 v1, so local infill contexts are not refreshed after accepted edits; this remains a stated limitation.

A Gate Q PASS does **not** authorize PPO implementation or training. It only permits drafting a new three-head PPO specification for separate expert approval. `final_test` remains sealed.
