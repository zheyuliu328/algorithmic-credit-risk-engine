"""Train-only preprocessing, recomputable metrics and honest CLI outcomes."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_one import run, synthetic_demo


def test_metrics_use_predictions():
    labels = [0, 0, 1, 1]
    metrics = synthetic_demo.classification_metrics(labels, [0.1, 0.4, 0.35, 0.8])
    assert metrics == pytest.approx({"auc": 0.75, "ks": 0.5, "brier": 0.158125})
    changed = synthetic_demo.classification_metrics(labels, [0.1, 0.2, 0.8, 0.9])
    assert changed == pytest.approx({"auc": 1.0, "ks": 1.0, "brier": 0.025})


@pytest.mark.parametrize("labels,scores", [
    ([0, 0], [0.1, 0.2]), ([0, 1], [0.1, float("nan")]),
    ([0, 1], [0.1, 1.1]), ([0, 1], [0.1]),
])
def test_metrics_reject_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        synthetic_demo.classification_metrics(labels, scores)


def test_reproduction_and_seed_change():
    first = synthetic_demo.run_experiment(seed=42, n_samples=400)
    assert first == synthetic_demo.run_experiment(seed=42, n_samples=400)
    changed = synthetic_demo.run_experiment(seed=43, n_samples=400)
    assert first["held_out"] != changed["held_out"]
    assert first["metrics"] != changed["metrics"]


def test_metrics_recompute_from_report_and_train_only_scaler():
    report = synthetic_demo.run_experiment(seed=42, n_samples=400)
    features, labels = synthetic_demo.generate_dataset(seed=42, n_samples=400)
    train = report["split"]["train_indices"]
    test = report["split"]["test_indices"]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(400))
    np.testing.assert_allclose(report["fit"]["scaler_mean"], features[train].mean(axis=0))
    assert report["fit"]["scaler_fit_samples"] == len(train)
    rows = report["held_out"]
    assert [row["sample_index"] for row in rows] == test
    assert [row["label"] for row in rows] == labels[test].tolist()
    actual = synthetic_demo.classification_metrics(
        [row["label"] for row in rows], [row["probability"] for row in rows]
    )
    assert actual == report["metrics"]


def test_heldout_feature_changes_cannot_change_fitted_parameters(monkeypatch):
    report = synthetic_demo.run_experiment(seed=42, n_samples=400)
    features, labels = synthetic_demo.generate_dataset(seed=42, n_samples=400)
    features[report["split"]["test_indices"]] += 100.0
    monkeypatch.setattr(synthetic_demo, "generate_dataset", lambda seed, n_samples: (features, labels))
    changed = synthetic_demo.run_experiment(seed=42, n_samples=400)
    assert changed["fit"] == report["fit"]
    assert changed["held_out"] != report["held_out"]


def test_demo_cli_from_unrelated_directory(tmp_path):
    first = tmp_path / "first" / "report.json"
    second = tmp_path / "second" / "report.json"
    for output in (first, second):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src/credit_one/run.py"), "demo", "--seed", "7",
             "--samples", "400", "--output", str(output)],
            cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Held-out AUC=" in result.stdout
    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text())
    assert report["seed"] == 7
    assert report["split"]["test_samples"] == 100
    assert report["mode"] == "SYNTHETIC_CLASSIFICATION_DEMO"


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_validate_is_nonzero_and_does_not_write(tmp_path, extra):
    output = tmp_path / "never-created" / "validation.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/credit_one/run.py"), "validate", "--output", str(output), *extra],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "NOT IMPLEMENTED" in result.stderr
    assert "Validation complete" not in result.stdout
    assert not output.parent.exists()


def test_invalid_seed_fails_without_output(tmp_path):
    output = tmp_path / "invalid.json"
    assert run.main(["demo", "--seed", "-1", "--output", str(output)]) != 0
    assert not output.exists()


def test_dashboard_uses_actual_app_and_propagates_failure(monkeypatch):
    calls = []

    def fake_run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run.main(["dashboard", "--port", "8502"]) == 7
    assert calls == [[sys.executable, "-m", "streamlit", "run",
                      str(ROOT / "src/credit_one/app.py"), "--server.port", "8502"]]
