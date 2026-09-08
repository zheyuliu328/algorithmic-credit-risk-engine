"""Regressions for destructive checks, overwritten evidence and misleading validation."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_one import model_validation, run
from credit_one.sme_credit_explainability import SMEConfig, generate_synthetic_sme_data
from scripts import run_real, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", [1, 5, 6, 100, np.int64(17)])
def test_generator_honors_requested_row_count(size):
    data = generate_synthetic_sme_data(n_samples=size)
    assert len(data) == size
    assert not data.isna().any().any()
    assert data["company_id"].is_unique


def test_generator_default_is_backward_compatible():
    pd.testing.assert_frame_equal(
        generate_synthetic_sme_data(), generate_synthetic_sme_data(SMEConfig.N_SAMPLES)
    )


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "100", np.nan])
def test_generator_rejects_invalid_sample_sizes(size):
    with pytest.raises(ValueError, match="positive integer"):
        generate_synthetic_sme_data(size)


def test_demo_refuses_to_overwrite_evidence(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_bytes(b"existing evidence")

    def never_fit(*args, **kwargs):
        raise AssertionError("An existing destination should be rejected before fitting")

    from credit_one import synthetic_demo

    monkeypatch.setattr(synthetic_demo, "run_experiment", never_fit)
    assert run.main(["demo", "--output", str(output)]) != 0
    assert output.read_bytes() == b"existing evidence"


def test_csv_reruns_keep_existing_files(tmp_path):
    source = ROOT / "data/sample_input.csv"
    first = run_real.run_scoring(source, tmp_path)
    evidence = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    second = run_real.run_scoring(source, tmp_path)
    assert first["run_id"] != second["run_id"]
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert len(list(tmp_path.glob("*.csv"))) == 2
    for name, content in evidence.items():
        assert (tmp_path / name).read_bytes() == content


def test_csv_run_id_collision_does_not_replace_evidence(tmp_path, monkeypatch):
    from datetime import datetime

    fixed_time = datetime(2026, 9, 8, 12)
    monkeypatch.setattr(run_real, "datetime", SimpleNamespace(now=lambda: fixed_time))
    monkeypatch.setattr(run_real, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    source = ROOT / "data/sample_input.csv"
    run_real.run_scoring(source, tmp_path)
    evidence = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        run_real.run_scoring(source, tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == evidence


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_csv_rejects_nonfinite_inputs_before_output(tmp_path, value):
    data = pd.read_csv(ROOT / "data/sample_input.csv")
    data.loc[0, "dti"] = value
    source = tmp_path / "invalid.csv"
    data.to_csv(source, index=False)
    output = tmp_path / "never-created"
    assert run_real.main([str(source), "--output", str(output)]) != 0
    assert not output.exists()


def test_csv_rejects_empty_input_before_output(tmp_path):
    data = pd.read_csv(ROOT / "data/sample_input.csv").iloc[:0]
    source = tmp_path / "empty.csv"
    data.to_csv(source, index=False)
    output = tmp_path / "never-created"
    assert run_real.main([str(source), "--output", str(output)]) != 0
    assert not output.exists()


def test_standalone_validation_fails_without_fabricating_metrics(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/credit_one/model_validation.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "NOT IMPLEMENTED" in result.stderr
    assert "AUC" not in result.stdout
    assert not list(tmp_path.iterdir())


def test_cap_trapezoids_use_current_numpy_compatible_integration():
    # The existing CAP definition omits an origin point; test that definition,
    # without upgrading this legacy metric utility into a validation claim.
    validator = model_validation.ModelValidator("fixture", "test")
    result = validator.cap_curve_analysis(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert np.isfinite(result["accuracy_ratio"])
    assert result["accuracy_ratio"] == pytest.approx(1.0)


def test_verify_rejects_existing_output_without_touching_it(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("preserve me")
    assert verify.main(["--output-dir", str(tmp_path)]) != 0
    assert evidence.read_text() == "preserve me"
    assert list(tmp_path.iterdir()) == [evidence]


def test_verify_propagates_child_failure(tmp_path, monkeypatch, capsys):
    def fail(_output):
        raise subprocess.CalledProcessError(23, ["failed-check"])

    monkeypatch.setattr(verify, "verify", fail)
    assert verify.main(["--output-dir", str(tmp_path / "new")]) == 23
    assert "CHECK FAILED" in capsys.readouterr().err


def test_verify_checked_command_rejects_nonzero_process(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        verify.run_checked([sys.executable, "-c", "raise SystemExit(17)"], cwd=tmp_path, env=None)
    assert failure.value.returncode == 17


def test_verify_stops_before_publishing_on_a_failed_check(tmp_path, monkeypatch):
    # Verify the reporting contract with explicit fake subprocess output; actual
    # subprocess behavior and complete real checks are exercised separately.
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(19, command)

    monkeypatch.setattr(verify.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        verify.verify(tmp_path)
    assert not (tmp_path / "verification.json").exists()


def test_legacy_metric_report_refuses_overwrite_and_disclaims_approval(tmp_path):
    validator = model_validation.ModelValidator("fixture", "test")
    output = tmp_path / "metrics.json"
    report = validator.generate_validation_report(output)
    assert report["validation_framework"] == "Educational metric checks; not regulatory approval"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        validator.generate_validation_report(output)
    assert output.read_bytes() == before
