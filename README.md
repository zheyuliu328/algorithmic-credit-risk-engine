# Algorithmic Credit Risk Engine

Educational credit-scoring prototypes, metric utilities and data-processing examples. This portfolio is not a deployed, independently validated or regulatory-compliant credit decision system.

## Install and run

Python 3.9+ is required; the complete supported checks were exercised on Python 3.9 and 3.12. Install from the repository root in a virtual environment:

```bash
python -m pip install ".[dev]"

# Fit on synthetic training rows and evaluate a separate 25% holdout.
credit-one demo --seed 42 --samples 1000 --output artifacts/demo_report.json

# Apply the documented DTI heuristic to the bundled three-row example.
python scripts/run_real.py data/sample_input.csv --output artifacts/csv_demo
```

The installed `credit-one` command works outside the checkout. The source entry point remains `python src/credit_one/run.py`. Reusing an existing demo-report filename fails without overwriting it. CSV runs receive unique IDs and create new files; an output collision also fails. Empty and non-finite CSV inputs are rejected before output is created.

Installation can access package indexes. The two examples above do not call a remote data service. Neither example is calibrated to borrower defaults or suitable for credit decisions.

## What is implemented

| Component | Supported scope |
| --- | --- |
| [Synthetic classification experiment](src/credit_one/synthetic_demo.py) | Independent normal features, logistic latent probabilities and sampled labels; train-only scaling and fitted logistic regression. Held-out AUC, two-sided KS and Brier loss are recomputable from saved predictions. |
| [SME prototype](src/credit_one/sme_credit_explainability.py) | Configurable synthetic row count, boosting/SHAP components and a WoE/logistic scorecard. Complete tests exercise scoring, score range and a debt-ratio comparison. Legacy demonstration adjustments remain disclosed. |
| [Metric utilities](src/credit_one/model_validation.py) | Metrics from caller-supplied labels and predictions. Reports refuse existing destinations and state that they are educational checks. The old standalone label-derived prediction demo has been disabled. |
| [CSV heuristic](scripts/run_real.py) | Required-column and numeric checks, followed by an unchanged debt-to-income formula, explicitly labelled `heuristic_dti_formula_v1`. It does not fit or load XGBoost. |
| [Optional dashboard](src/credit_one/app.py) | Experimental Streamlit interface. Install with `python -m pip install ".[dashboard]"`; launch with `credit-one dashboard`. Online actions and the UI are outside the offline acceptance suite. |
| [Macro/ECL research](src/credit_one/ecl/) | Separate research modules. Dedicated ECL tests and CLI/dashboard integration remain incomplete in the [feature checklist](feature_checklist.json). |

The synthetic experiment saves its generation rule, seed, train/test indices, fitted scaling parameters, library versions and held-out labels/probabilities. A constant training-event-rate predictor provides a reference. No seed or hyperparameter search is performed. A random IID holdout does not test future economic periods, calibration, fairness or deployment suitability.

## Verification

```bash
python -m pytest tests/ -v
make lint
python -m black --check .
make verify
```

On 2026-09-08, all **54 tests passed with zero skips** in isolated Python 3.9.25 and 3.12.12 environments. Ruff and Black passed without changing files. Wheel/editable installation, safe output behavior and the full verification entry point are recorded in [CI revalidation](docs/CI_REVALIDATION.md).

`make verify` runs the complete tests, code checks, configuration validation, fresh offline examples and wheel construction. It writes new evidence to a fresh temporary directory and never clears existing artifacts. Failures propagate as nonzero process statuses. With `python scripts/verify.py --output-dir <new-directory>`, an existing destination is rejected.

The [CI workflow](.github/workflows/ci.yml) retains its `lint`, `test`, `e2e` and `verify` jobs. Its checks do not auto-fix files or suppress failures. The independent [Security workflow](.github/workflows/security.yml) retains the Git-history gitleaks scan. These local results are not a claim that a not-yet-published revision has passed remote Actions.

The separate [Synthetic CLI checks](.github/workflows/synthetic-demo.yml) workflow continues to exercise the smaller offline subset. It is not a replacement for the full suite.

## Boundaries that remain

- `validate`, including `validate --dry-run`, exits 2 without reporting success. The standalone `model_validation.py` entry also exits 2 rather than constructing predictions from labels. A supplied-model validation workflow is still required.
- Legacy SME/VIP and large-company adjustments are demonstration rules. Their adjusted probabilities must not be used as untouched model-evaluation evidence.
- The existing PSI implementation uses finite baseline quantile endpoints. Out-of-range observations can fall outside its histogram, and its fixed 0.1/0.25 interpretation bands are heuristics. Those model-methodology limitations were not hidden by the CI repair.
- Legacy CAP/Gini and assessment thresholds remain research utilities; their historical definitions were preserved. Passing software tests does not certify those definitions or financial conclusions.
- The Lending Club loader still refers to a machine-specific cache path. Approximate macro tables are demonstrations, not verified FRED downloads; credit-card delinquency is not borrower default incidence.
- Optional online data access, dashboard rendering, chronological validation and complete ECL methodology remain outside this acceptance scope.

See [Portfolio status](docs/PORTFOLIO_STATUS.md), [limitations](docs/limitations.md), [CSV format](docs/real-data.md) and the [MIT code license](LICENSE). Historical architecture/quickstart documents may retain old paths or aspirational examples; this README and the dated revalidation record describe current acceptance.

For a separate experiment using fully synthetic time-series data, see [model-risk-lab](https://github.com/zheyuliu328/model-risk-lab). Its results do not retroactively validate this repository.
