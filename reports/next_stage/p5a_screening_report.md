# P5a Development Screening

- Gate: **PASS**
- Sources: 170 (10/device), dual-clean-correct development data
- Policy Gate / final test used: **false / false**
- Candidate Victim queries: 35671
- Greedy minus best random at 8 edits: 11.32 pp
- Retained methods: `greedy_infill, random_infill, random_nearest`

Primary 8-edit results:

| method | query setting | valid-hard | B retention | RF valid |
|---|---:|---:|---:|---:|
| random_uniform_plausible | q0 | 0.000 | 1.000 | 1.000 |
| random_nearest | q0 | 0.000 | 1.000 | 1.000 |
| random_infill | q0 | 0.000 | 1.000 | 1.000 |
| greedy_infill | q256 | 0.113 | 1.000 | 1.000 |
