# P5b Formal Paired Policy-Gate Experiment

- Gate: **PASS**
- Sources: 425 (25/device), fixed untouched-buffer policy Gate
- Codec-joint-correct denominator: 412
- Final test used: **false**
- Candidate Victim queries: 88599

| method | Q | edits | valid-hard | 95% CI | B retention | RF valid |
|---|---:|---:|---:|---:|---:|---:|
| random_nearest | 0 | 2 | 0.004 | [0.000, 0.011] | 1.000 | 1.000 |
| random_nearest | 0 | 4 | 0.005 | [0.000, 0.011] | 1.000 | 1.000 |
| random_nearest | 0 | 8 | 0.005 | [0.001, 0.010] | 1.000 | 1.000 |
| random_nearest | 0 | 16 | 0.006 | [0.002, 0.011] | 1.000 | 1.000 |
| random_infill | 0 | 2 | 0.003 | [0.000, 0.010] | 1.000 | 1.000 |
| random_infill | 0 | 4 | 0.002 | [0.000, 0.005] | 1.000 | 1.000 |
| random_infill | 0 | 8 | 0.004 | [0.001, 0.009] | 1.000 | 1.000 |
| random_infill | 0 | 16 | 0.005 | [0.002, 0.009] | 1.000 | 1.000 |
| greedy_infill | 64 | 2 | 0.034 | [0.017, 0.053] | 1.000 | 1.000 |
| greedy_infill | 64 | 4 | 0.041 | [0.022, 0.061] | 1.000 | 1.000 |
| greedy_infill | 64 | 8 | 0.058 | [0.036, 0.080] | 1.000 | 1.000 |
| greedy_infill | 64 | 16 | 0.070 | [0.046, 0.095] | 1.000 | 1.000 |
| greedy_infill | 128 | 2 | 0.041 | [0.024, 0.061] | 1.000 | 1.000 |
| greedy_infill | 128 | 4 | 0.066 | [0.044, 0.090] | 1.000 | 1.000 |
| greedy_infill | 128 | 8 | 0.090 | [0.063, 0.119] | 1.000 | 1.000 |
| greedy_infill | 128 | 16 | 0.117 | [0.085, 0.148] | 1.000 | 1.000 |
| greedy_infill | 256 | 2 | 0.063 | [0.041, 0.087] | 1.000 | 1.000 |
| greedy_infill | 256 | 4 | 0.107 | [0.078, 0.136] | 1.000 | 1.000 |
| greedy_infill | 256 | 8 | 0.148 | [0.114, 0.182] | 1.000 | 1.000 |
| greedy_infill | 256 | 16 | 0.182 | [0.146, 0.221] | 1.000 | 1.000 |
