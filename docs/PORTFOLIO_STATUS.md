# Portfolio status: credit-risk prototypes

Audit baseline: 2026-09-07. This record distinguishes source-level functionality, isolated smoke tests, and work that still needs verification. It is not a certification or model-validation approval.

## Scope and observed runs

Updated on 2026-09-07 after the offline CLI repair. The implementation was built from a fresh synthetic generator and did not reuse legacy scoring functions. Tests and demonstrations wrote to temporary directories; no external dataset, network request or dependency installation was used. Destructive verification scripts were not run.

| Entry point / check | Observed result | What it establishes |
| --- | --- | --- |
| `python src/credit_one/run.py demo --seed 42 --samples 1000 --output <new-file>` | Exit 0; 1,000 synthetic rows, 750 training and 250 held-out rows | A fitted StandardScaler/LogisticRegression pipeline produces measured AUC, two-sided KS separation and binary Brier loss, plus labels/predictions for recomputation. |
| `python scripts/run_real.py data/sample_input.csv --output <new-directory>` | Exit 0; three rows and fresh CSV/JSON artifacts | The existing DTI heuristic is unchanged; report identity is corrected to `heuristic_dti_formula_v1`. |
| `validate`, including `validate --dry-run` | Exit 2; explicit unimplemented message; no output file/directory created | Failure is reported honestly. No model-validation workflow is claimed. |
| Dashboard launcher test | Correct source path/current interpreter; simulated child failure returns the same nonzero status | Launcher wiring and status propagation only. The optional Streamlit UI was not launched. |
| Targeted tests | 16 passed: CLI demo, basic entry points and CSV E2E | Covers seed reproducibility, manually checkable metric values, report recomputation, train-only scaling and fresh output paths. |
| Legacy `tests/test_credit_model.py` collection | Fails with `ModuleNotFoundError: sme_credit_explainability` | The complete legacy test suite is not passing. This failure is not hidden by the targeted result. |

Test environment: Python 3.14.5, NumPy 2.4.2, scikit-learn 1.8.0. SHAP, optbinning and Streamlit were absent in this environment; their optional workflows were not validated. Numeric reproduction across environments should consider the recorded library versions.

## New experiment protocol

The [new generator](../src/credit_one/synthetic_demo.py) draws four independent standard-normal features. Arbitrary coefficients `(0.8, -0.6, 0.4, 0.0)` and intercept `-1.0` define latent logistic event probabilities; labels are Bernoulli draws. These are simulated binary events, not calibrated credit outcomes.

A seeded stratified 75/25 split is made before fitting. StandardScaler and LogisticRegression are fitted only on training rows. There is no seed, feature or hyperparameter search. A constant predictor based solely on the training event rate provides a reference. The JSON report includes the generation rule, seed, sample counts, indices, fitted parameters, held-out labels/probabilities, metric definitions and software versions.

KS is the maximum absolute difference between TPR and FPR along the ROC curve; it describes separation, not probability calibration. Brier is the mean squared error between binary labels and probabilities. AUC and KS do not establish credit calibration. The IID random holdout does not test economic time stability or out-of-time performance.

The preprocessing test changes only held-out features and verifies that fitted model/scaler parameters remain unchanged. Separate tests use hand-checkable predictions to detect hard-coded metrics and rerun the CLI from an unrelated working directory.

## Repairs and remaining gaps

| Topic | Current state | Remaining acceptance criterion |
| --- | --- | --- |
| Fixed default-demo metrics | Replaced by the [fresh fitted experiment](../src/credit_one/synthetic_demo.py) | Do not use historical hard-coded report values as evidence for this implementation or real credit performance. |
| Validation CLI | [CLI](../src/credit_one/run.py) now rejects the unimplemented operation with exit 2, including dry-run | Implement a genuine supplied-model evaluation workflow and assert its current-run report before enabling success. |
| CSV model identity | [CSV script](../scripts/run_real.py) identifies its unchanged DTI formula as a heuristic | A fitted-model CSV route would require a separate model contract, version and evaluation. |
| Dashboard path | Uses the actual source file and propagates the subprocess status | Install and test optional UI dependencies separately; market-data actions are outside offline acceptance. |
| Basic/E2E tests | [Basic tests](../tests/test_basic.py) assert actual callable/help behavior; [E2E test](../tests/test_e2e.py) uses a fresh temporary directory and checks contents | Repair the remaining legacy tests; do not call a subset a fully passing suite. |
| Legacy scoring tests | Import path is broken; inspected callers pass `n_samples` to an older generator that accepts no arguments | Align imports and signatures; exercise scorecard behavior with its required dependencies. |
| Standalone OOT illustration | [Old metric module](../src/credit_one/model_validation.py) constructs predictions directly from labels; its earlier isolated run printed AUC 1.0000 and then failed at `np.trapz` | Treat as a metric illustration. Fix compatibility and use genuine fitted/chronological predictions before claiming OOT validation. |
| Legacy prediction overrides | [Older SME module](../src/credit_one/sme_credit_explainability.py) adjusts selected demonstration records and returned probabilities | Separate presentation adjustments from estimator outputs and performance evaluation. |
| Compliance wording / pipeline maturity | Older modules use production/compliance terminology; [limitations](limitations.md) reject those claims | No deployment, regulatory approval or complete legacy validation was demonstrated. |

The [ECL checklist](../feature_checklist.json) leaves dedicated tests and CLI/dashboard integration unfinished. Its historical figures were not reproduced and are not validation evidence for either the old modules or the new experiment.

## Data provenance

| Data route | Evidence in this repository | Boundary / missing evidence |
| --- | --- | --- |
| Fresh offline experiment | [`synthetic_demo.py`](../src/credit_one/synthetic_demo.py) | Independent normal inputs and arbitrary logistic/Bernoulli simulation; no external source or credit calibration. |
| Legacy synthetic SME records | [`generate_synthetic_sme_data`](../src/credit_one/sme_credit_explainability.py) | Synthetic labels and demonstration adjustments; not a client portfolio or representative credit population. |
| Bundled CSV example | [`sample_input.csv`](../data/sample_input.csv) and [CSV guide](real-data.md) | Three input rows, no outcome labels. Document them as toy inputs, not a validation dataset. |
| Lending Club | [`load_lending_club_data.py`](../src/credit_one/load_lending_club_data.py) and [`pipeline.py`](../src/credit_one/pipeline.py) name the `wordsforthewise/lending-club` Kaggle dataset | Loader uses a machine-specific path. Dataset access, version identity, licensing and complete reproducibility were not independently verified. |
| Approximate macro bundle | [`generate_macro_data.py`](../scripts/generate_macro_data.py) explicitly says its manually listed values approximate historical FRED patterns | This generator does not download official observations. The bundle must be labelled approximate demonstration data unless its provenance is separately established. |
| FRED API route | [`macro_data.py`](../src/credit_one/ecl/macro_data.py) names GDP, UNRATE, FEDFUNDS and DRCCLACBS series | API capability is separate from the approximate bundle. Record retrieval time, vintage, series units and frequency conversion for any actual download. |
| Target proxy | The same module maps the credit-card delinquency series to `observed_default_rate` | Delinquency and default are different concepts. State the proxy explicitly; do not treat it as observed default incidence or validated ECL input. |
| Yahoo Finance | [`predict_from_live_data`](../src/credit_one/sme_credit_explainability.py) | Optional online inputs; availability, timing, proxy transformations and data-use terms require separate verification. |

The code license does not establish third-party dataset redistribution rights. Add source URLs, exact dataset/series identifiers, retrieval dates, licenses/terms, transformations and checksums before distributing a claimed reproducible external-data snapshot.

## Reproducible next steps

- Keep offline examples independent of credentials and online fallback behavior.
- Implement the currently disabled validation workflow; separately exercise the optional UI.
- Align optional dependencies, supported Python versions and legacy test imports/signatures; pass the complete suite.
- Keep fabricated predictions, heuristic formulas and trained estimates visibly distinct.
- Retain reproducible synthetic results and reference comparisons; obtain suitable evidence before making any real-data performance claim.
- Treat macro/ECL experiments as unvalidated research until their data definitions, temporal evaluation and tests are complete.

The separate [model-risk-lab](https://github.com/zheyuliu328/model-risk-lab) provides the new direction for a fully synthetic regression-validation portfolio. It does not retroactively validate this repository.
