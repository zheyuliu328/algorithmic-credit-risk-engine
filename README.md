# Algorithmic Credit Risk Engine

Educational credit-scoring prototypes, model-metric utilities, and data-processing examples. The repository is a portfolio demonstration; it is not a deployed, independently validated, or regulatory-compliant credit decision system.

For a separate validation experiment built from fully synthetic time-series data, see [model-risk-lab](https://github.com/zheyuliu328/model-risk-lab). The existing modules here are not evidence of that project's validation results.

## What is implemented

| Component | Scope |
| --- | --- |
| [Offline classification experiment](src/credit_one/synthetic_demo.py) | Fresh independent normal features, logistic latent probabilities, sampled binary labels, train-only scaling and a fitted logistic-regression baseline. Reports recomputable held-out AUC, KS and Brier loss. |
| [SME scoring prototype](src/credit_one/sme_credit_explainability.py) | Synthetic-data generation, a boosting classifier, optional WoE/logistic scorecard, score scaling, and SHAP integration. These are experimental components, not calibrated production PDs. |
| [Metric utilities](src/credit_one/model_validation.py) | Functions for AUC, K-S, PSI, CAP/Gini and calibration summaries from supplied labels and predictions. The standalone example fabricates predictions from labels; it does not validate a trained model. |
| [CSV demonstration](scripts/run_real.py) | Input-column checks and a simple debt-to-income scoring formula. It does not load or train an XGBoost model. |
| [Streamlit interface](src/credit_one/app.py) | Experimental scoring and drift-monitoring interface. The interface and its optional dependencies were not included in the offline smoke test. |
| [Macro/ECL experiments](src/credit_one/ecl/) | Separate research modules. Their bundled values are approximate demonstrations, and the complete pipeline has not been independently validated. |

## Offline examples

Run from the repository root with Python 3.9+ and NumPy, scikit-learn and pandas already installed. Neither command below calls a remote data service. Dependency installation is a separate setup step.

```bash
# Fit once on synthetic training rows; evaluate a separate 25% holdout.
python src/credit_one/run.py demo --seed 42 --samples 1000 --output artifacts/demo_report.json

# Requires pandas: checks the bundled three-row CSV and applies a DTI formula.
python scripts/run_real.py data/sample_input.csv --output artifacts/csv_demo
```

The first report records the seed, generation rule, train/test indices, fitted scaling parameters, software versions and held-out labels/probabilities. Its AUC, two-sided KS separation and Brier loss are calculated from those predictions. A constant training-event-rate predictor is included for comparison; no seed or parameter search is performed.

The features and event probabilities have no credit-business calibration. The random holdout evaluates IID simulated rows, not future economic periods. The second command is separate: it applies the unchanged DTI heuristic and identifies it as `heuristic_dti_formula_v1`, not XGBoost.

For the optional interface, the source-file location is:

```bash
python src/credit_one/run.py dashboard --port 8501
```

The launcher invokes the actual `src/credit_one/app.py` with the current Python interpreter and propagates its process exit status. Install optional interface dependencies from `requirements.txt` separately. Market-data actions may access Yahoo Finance. The UI itself was not smoke-tested.

## Verification status

On 2026-09-07, the synthetic experiment and CSV example ran with isolated output paths. Sixteen targeted tests passed, covering recomputable metrics, seed reproducibility, train-only preprocessing, CLI behavior and fresh CSV artifacts. This is a measured synthetic experiment, not evidence of real credit-model performance.

The [Synthetic CLI checks](https://github.com/zheyuliu328/algorithmic-credit-risk-engine/actions/workflows/synthetic-demo.yml) workflow runs this targeted suite and publishes a fresh report. It is separate from the legacy full-project workflow.

`validate` remains unimplemented and now exits nonzero without producing a report, including with `--dry-run`. Legacy scorecard tests still fail during import, and their generator arguments need alignment. The old standalone metric example also has a NumPy compatibility failure. See [Portfolio status](docs/PORTFOLIO_STATUS.md) for the remaining gaps.

```bash
python -m pytest -p no:cacheprovider tests/test_cli_demo.py tests/test_e2e.py tests/test_basic.py
```

Fixed AUC, accuracy, training-time and inference-time tables from older documentation should not be read as measured results. Historical passing badges and legacy examples are not a substitute for a reproducible test run.

## Data and methodology boundaries

- The new offline experiment uses a separate generator with no special-case IDs or prediction overrides. The older SME generator still contains demonstration adjustments; its returned predictions are not an untouched evaluation output.
- `data/sample_input.csv` is a tiny bundled example without outcome labels. It cannot support an AUC or model-validation conclusion.
- The legacy Lending Club loader refers to a public Kaggle dataset through a machine-specific cache path. That path is not portable; external data retrieval and licensing were not verified in this audit.
- The macro-data generator describes its table as approximate values based on historical FRED patterns. Do not describe that bundle as a verified FRED download. Its credit-card delinquency proxy is not a measured borrower default-rate series.
- Optional public-data integrations require their own source, version, retrieval and usage documentation. The repository's code license does not establish rights to redistribute third-party data.

No production, regulatory-compliance, credit-approval or investment-performance claim follows from these examples. [Project limitations](docs/limitations.md) and [legal scope](docs/legal.md) remain applicable.

## Next acceptance milestones

1. Implement a genuine supplied-model validation workflow before enabling `validate`.
2. Repair legacy imports/generator signatures and align optional dependencies; pass the complete test suite in a clean environment.
3. Keep the legacy metric illustrations and adjusted predictions distinct from fitted-model evaluation.
4. Establish external-data provenance and suitable validation before publishing any real-data performance claim.

## Navigation

- [Portfolio status and audit evidence](docs/PORTFOLIO_STATUS.md)
- [Source modules](src/credit_one/)
- [CSV input format](docs/real-data.md)
- [Limitations](docs/limitations.md)
- [MIT code license](LICENSE)

Start with this page and the status record. Other architecture and quickstart documents contain historical examples and may use old paths or aspirational descriptions.
