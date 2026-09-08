# CI and installation revalidation — 2026-09-08

Baseline: `c0ca2a58f918eea3b821b4faa4bb6bb58cd5ac62`. Changes were prepared in an isolated copy and tested with normal public dependency installation. Existing data, model artifacts and images were not modified. The local acceptance below precedes publication; remote results for each revision are recorded by the [CI workflow](https://github.com/zheyuliu328/algorithmic-credit-risk-engine/actions/workflows/ci.yml).

## Baseline failures reproduced

- The [baseline CI run](https://github.com/zheyuliu328/algorithmic-credit-risk-engine/actions/runs/34139071251) installed dependencies successfully but failed in Ruff: 353 findings, 228 auto-fixed and 125 remaining. `test`, `e2e` and `verify` were skipped downstream.
- Wheel and editable builds failed because Hatch could not infer a package named `algorithmic_credit_risk_engine`; the actual package is `src/credit_one`.
- Full pytest failed during the legacy top-level import. With only the import search path supplied for diagnosis, 22 tests passed and 5 failed: four unsupported `n_samples` calls and one incorrect random-sample PSI bound.
- The seeded PSI sample has baseline counts `[100] * 10` and current counts `[36, 49, 48, 70, 83, 76, 118, 117, 173, 230]`. An independent sum of `(q-p) * log(q/p)` gives `0.31432662909735887`; all 1,000 observations enter the bins. An asserted upper bound of 0.3 was not justified for that finite sample.
- The old verification script could delete existing output and hide failed tests, lint or configuration checks.

## Changes and acceptance

Packaging explicitly includes `src/credit_one`; test imports use the package, and dependency declarations have one source of truth. The legacy generator now accepts validated sample counts while preserving its default generation rule. PSI arithmetic is unchanged. Tests instead use an independently counted oracle, a hand-calculable four-bin fixture and random-sample invariants. Scorecard tests no longer skip missing optbinning; the complete installation includes it.

Supported CLI outputs use exclusive file creation, and CSV runs have unique IDs. Invalid/non-finite or empty CSV input fails before writing. The old standalone fabricated-prediction demonstration is disabled. Legacy report metadata disclaims regulatory approval; a removed NumPy integration call is replaced by SciPy's same trapezoid rule.

CI retains all four job names and the separate gitleaks workflow. Ruff and Black check committed formatting; they do not auto-fix CI files or suppress status codes. `scripts/verify.py` executes all supported checks and preserves existing artifacts. Its failure propagation and refusal of an existing destination have dedicated regression tests.

## Local results

| Check | Result |
| --- | --- |
| Python 3.9.25 complete pytest | 54 passed, zero skipped |
| Python 3.12.12 complete pytest | 54 passed, zero skipped |
| Ruff 0.16.6 | Passed with the existing E/F/I/W rules |
| Black 25.11.0 | Passed; 27 Python files require no formatting changes |
| Normal package installation | Passed in isolated Python 3.9 and 3.12 environments |
| Editable installation | Passed with normal build isolation on Python 3.9 |
| Installed wheel outside the checkout | Passed on Python 3.12; import resolved to site-packages, CLI created a fresh report, repeated destination failed without changing bytes |
| Complete verification entry point | `make verify` passed on Python 3.9; the same seven real checks passed on Python 3.12 |
| Working-file gitleaks 8.30.0 | No findings |
| Git-history gitleaks 8.30.0 | No findings across 45 commits |

Selected installed versions:

| Package | Python 3.9 environment | Python 3.12 environment |
| --- | --- | --- |
| NumPy | 2.0.2 | 2.5.3 |
| pandas | 2.3.3 | 3.0.5 |
| scikit-learn | 1.6.1 | 1.9.0 |
| SciPy | 1.13.1 | 1.18.1 |
| SHAP | 0.49.1 | 0.52.0 |
| optbinning | 0.21.0 | 0.21.0 |
| pytest | 8.4.2 | 9.1.1 |
| statsmodels | 0.14.6 | 0.15.0 |

These are observed environments, not claims that every future dependency version is compatible. Optional dashboard/online extras were declared but were not included in the offline acceptance.

Commands from the repository root:

```bash
python -m pip install ".[dev]"
python -m pytest tests/ -q -p no:cacheprovider
python -m ruff check . --no-cache
python -m black --check .
python scripts/verify.py --output-dir <new-directory>
```

Verification builds a fresh wheel, validates its package contents, runs fresh synthetic/CSV examples and executes all tests. It retains `verification.json` and the generated evidence under the printed new directory. Dependency installation is a separate online step; verification does not download market data.

The macOS Python 3.9 run reports upstream numeric-library runtime warnings; Python 3.12 reports dependency deprecations. These are visible, not suppressed. Both suites completed successfully, but warning-free execution and cross-platform numerical identity are not claimed.

This acceptance covers software behavior and the current suite. It does not complete ECL-008/009/010, validate legacy PSI/CAP methodology, remove demonstration probability overrides, test the optional dashboard, or establish regulatory/production readiness.
