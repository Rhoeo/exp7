# exp7: RF Fingerprint Adversarial Sample Generation

This repository contains the source code, experiment configurations, audit reports, and progress records for a reference-conditioned RF fingerprint adversarial-sample generation study inspired by the structural ideas in AdvTG.

## Current status

Stages P0–P6 are complete. On the fixed untouched-buffer Policy Gate, the pre-registered Greedy score-based search passed the project Query Gate against a query-matched random plausible baseline:

- Q=256 conditional valid-hard rate: Greedy 14.81%, random 6.55%.
- Paired difference: +8.25 percentage points, 95% CI [5.83, 10.92].
- Greedy Q=128 reached 8.98%, exceeding random Q=256 while using half the query budget.
- Evaluator B retention and digital-domain RF-valid fraction were 100% for the reported P6 conditions.

These are conditional results on dual-clean-correct sources. They do not establish over-the-air feasibility or full-distribution attack rates. `final_test` remains sealed. PPO implementation/training is not authorized without a separately reviewed three-head PPO specification.

## Repository layout

- `rffi_core/`: data, model, token-graph, infill, RF-validity, and attack/search implementations.
- `configs/`: frozen data, generator, and attack configurations.
- `tests/`: unit tests for data splits, token graphs, infill, RF constraints, query search, and codec models.
- `reports/`: experiment reports, stage audits, diagnostic reports, and execution status.
- `runs/`: lightweight JSON histories and experiment metadata. Binary checkpoints and cached arrays are excluded.
- `artifacts/`: small reproducibility artifacts such as the token candidate graph.
- `defense_plan/`, `generative_llm_plan/`: planning and method-design documents.

## Key reports

- [`reports/next_stage/NEXT_STAGE_EXECUTION_STATUS_2026-08-29.md`](reports/next_stage/NEXT_STAGE_EXECUTION_STATUS_2026-08-29.md)
- [`reports/next_stage/p6_query_gate_report.md`](reports/next_stage/p6_query_gate_report.md)
- [`reports/next_stage/P6_DIAGNOSTIC_REPORT_2026-08-29.md`](reports/next_stage/P6_DIAGNOSTIC_REPORT_2026-08-29.md)
- [`reports/next_stage/NEXT_STAGE_IMPLEMENTATION_PLAN.md`](reports/next_stage/NEXT_STAGE_IMPLEMENTATION_PLAN.md)

## Verification

The local reference environment uses Python 3.10 and PyTorch. Run the unit suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The WiFi-B/ManyTx source datasets, local caches, paper PDFs, and model checkpoints are intentionally not stored in Git. Paths and checkpoint hashes needed for experiment auditing remain recorded in the reports.

