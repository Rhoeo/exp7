# P6 Query Search Gate

- Gate Q: **PASS**
- Gate Q1/Q2: **PASS / PASS**
- Sources: 425 (25/device), fixed untouched-buffer policy Gate
- Codec-joint-correct denominator: 412
- Edit budget: 8 tokens
- Final test used: **false**
- Evaluator B: offline final audit only

| method | Q cap | valid-hard | 95% CI | mean queries | B retention | min-device B | RF valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| query-matched random | 64 | 0.052 | [0.032, 0.073] | 64.0 | 1.000 | 1.000 | 1.000 |
| greedy-coordinate | 64 | 0.058 | [0.036, 0.083] | 53.8 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 64 | 0.012 | [0.002, 0.024] | 64.0 | 1.000 | 1.000 | 1.000 |
| query-matched random | 128 | 0.061 | [0.040, 0.084] | 128.0 | 1.000 | 1.000 | 1.000 |
| greedy-coordinate | 128 | 0.090 | [0.063, 0.119] | 105.4 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 128 | 0.017 | [0.005, 0.032] | 128.0 | 1.000 | 1.000 | 1.000 |
| query-matched random | 256 | 0.066 | [0.043, 0.089] | 256.0 | 1.000 | 1.000 | 1.000 |
| greedy-coordinate | 256 | 0.148 | [0.114, 0.182] | 208.5 | 1.000 | 1.000 | 1.000 |
| beam-coordinate | 256 | 0.029 | [0.015, 0.046] | 256.0 | 1.000 | 1.000 | 1.000 |

| Q | searcher | searcher - random (pp) | paired 95% CI (pp) | Gate Q1 |
|---:|---|---:|---:|---:|
| 64 | beam | -3.96 | [-5.83, -2.18] | FAIL |
| 64 | greedy | 0.65 | [-0.57, 1.94] | FAIL |
| 128 | beam | -4.45 | [-6.47, -2.67] | FAIL |
| 128 | greedy | 2.83 | [1.21, 4.61] | FAIL |
| 256 | beam | -3.64 | [-5.42, -2.10] | FAIL |
| 256 | greedy | 8.25 | [5.83, 10.92] | PASS |

## Interpretation boundary

The random baseline receives the same candidate pool, edit budget, projection, and per-source query cap. P5b zero-query random results are auxiliary and do not decide Gate Q. Candidate proposals are frozen from the clean reference during P6 v2, so local infill contexts are not refreshed after accepted edits; this remains a stated limitation.

A Gate Q PASS does **not** authorize PPO implementation or training. It only permits drafting a new three-head PPO specification for separate expert approval. `final_test` remains sealed.
