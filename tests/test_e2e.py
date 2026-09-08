"""Exercise the CSV heuristic with artifacts created only by the current run."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_real(tmp_path):
    output = tmp_path / "scoring"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_real.py"),
            str(ROOT / "data/sample_input.csv"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reports = list(output.glob("scoring_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["rows_processed"] == 3
    assert report["parameters"]["model"] == "heuristic_dti_formula_v1"
    assert reports[0].name == f"scoring_report_{report['run_id']}.json"
    csv_path = output / f"scoring_output_{report['run_id']}.csv"
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert math.isclose(float(rows[0]["pd_score"]), 0.1456)
    assert rows[0]["risk_grade"] == "B"
