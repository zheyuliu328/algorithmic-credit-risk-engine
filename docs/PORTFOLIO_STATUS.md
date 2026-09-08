# Portfolio status: credit-risk prototypes

Updated 2026-09-08. This record distinguishes executable software checks from financial-model validation and approval. Detailed commands and evidence are in [CI revalidation](CI_REVALIDATION.md).

| Area | Current result | Boundary |
| --- | --- | --- |
| Installation | Explicit `src/credit_one` wheel package; normal wheel and editable installation supported | External packages must be installed separately from offline execution. |
| Complete tests | 54 passed, zero skips on Python 3.9.25 and 3.12.12 | Covers the current test suite, not the unfinished ECL checklist. |
| Code checks | Ruff 0.16.6 and Black 25.11.0 pass; CI only checks | Formatting does not establish numerical or business correctness. |
| Synthetic classifier | Fitted train-only pipeline, separate holdout, measured AUC/KS/Brier and row-level predictions | IID simulation with arbitrary coefficients; no borrower calibration or real-data performance claim. |
| Legacy scorecard | Broken import and `n_samples` API repaired; actual scorecard tests execute with optbinning installed | Legacy special-case records and returned-probability overrides remain outside honest evaluation evidence. |
| PSI tests | Independent frequency-count calculation, deterministic moderate-shift fixture, random-sample/order invariants | The old random upper bound was invalid; the PSI algorithm was not changed to force that sample below a threshold. Finite-tail and threshold-policy limitations remain. |
| Output safety | Demo/report destinations are exclusively created; CSV run IDs are unique and collisions fail | Existing CSV, model and image assets remain untouched. |
| Verification | Complete tests and required local checks propagate failure; evidence uses a new directory | No cleanup of existing artifacts. Security Actions is a separate check, not inferred from a config file. |
| Validation CLI | `validate` and standalone `model_validation.py` explicitly exit 2 | Supplied-model validation is not implemented; no label-derived fake predictions are generated. |
| Optional dashboard/data sources | Dependencies are declared under extras | UI and online-source behavior were not accepted by the offline suite. |
| Macro/ECL | Research modules retained, with formatting-only changes | ECL-008/009/010 remain incomplete. Historical figures were not reproduced or promoted to validation evidence. |

## Data and model boundaries

The fresh synthetic classifier does not reuse customer data, legacy predictions or special-case IDs. It fits scaling and logistic regression using only training rows. Reports record their generation and split rules so metrics can be recalculated. It is an IID learning exercise, not chronological validation.

The three-row `data/sample_input.csv` has no outcome labels and cannot establish AUC or PD calibration. The CSV route applies the documented DTI heuristic and is not XGBoost.

The old SME module contains VIP and large-company adjustments. The separate legacy CAP/Gini calculations and PASS/WARNING thresholds have not been validated as regulatory metrics. A compatibility replacement for removed NumPy `trapz` uses the same trapezoid rule; it does not endorse the surrounding methodology. Report metadata now states its educational scope rather than claiming SR 11-7 compliance.

Approximate macro tables do not establish actual FRED retrieval. The credit-card delinquency proxy is not measured borrower default incidence. External datasets need source/version/retrieval/licensing evidence before any redistribution or real-data performance claim.

## Remaining work

1. Design and implement a genuine supplied-model validation contract, with chronological evaluation and explicit metric definitions.
2. Review PSI tail handling, sparse-bin smoothing and thresholds separately; test the methodology before changing it.
3. Separate all legacy presentation adjustments from estimator outputs and performance evaluation.
4. Complete dedicated ECL tests and integration before marking the corresponding checklist items complete.
5. Verify optional UI behavior and external-data provenance independently.

A passing software pipeline does not imply production readiness, regulatory approval, credit approval or independently validated financial models.
