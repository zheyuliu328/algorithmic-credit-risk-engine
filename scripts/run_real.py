#!/usr/bin/env python3
"""
Educational CSV scoring with an explicit debt-to-income heuristic.
This entry point does not fit a model or validate credit performance.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def validate_csv(csv_path: str) -> dict:
    """验证输入 CSV 格式"""
    required_columns = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "earliest_cr_line",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
    ]

    if not os.path.exists(csv_path):
        return {"valid": False, "error": f"File not found: {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"valid": False, "error": f"Cannot read CSV: {e}"}

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return {"valid": False, "error": f"Missing columns: {missing}"}

    if df.empty:
        return {"valid": False, "error": "CSV must contain at least one input row"}

    # Validate the fields used by the supported heuristic.
    errors = []
    for col in ["loan_amnt", "annual_inc", "dti"]:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            errors.append(f"{col} must be numeric")
        elif not np.isfinite(df[col].to_numpy(dtype=float)).all():
            errors.append(f"{col} must contain finite values")

    if errors:
        return {"valid": False, "error": "; ".join(errors)}

    return {"valid": True, "rows": len(df), "columns": list(df.columns)}


def run_scoring(csv_path: str, output_dir: str = "artifacts") -> dict:
    """运行评分流程"""
    # Validate input
    validation = validate_csv(csv_path)
    if not validation["valid"]:
        raise ValueError(f"Validation failed: {validation['error']}")

    print(f"[INFO] Validated {validation['rows']} rows")

    # Load and process (simplified for demo)
    df = pd.read_csv(csv_path)

    # Apply the documented heuristic without claiming a fitted PD model.
    df["pd_score"] = 0.1 + 0.3 * (df["dti"] / 100).clip(0, 1)
    df["risk_grade"] = df["pd_score"].apply(
        lambda x: "A" if x < 0.1 else "B" if x < 0.2 else "C" if x < 0.3 else "D"
    )

    # Generate report
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    report = {
        "run_id": run_id,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(csv_path),
        "rows_processed": len(df),
        "parameters": {"model": "heuristic_dti_formula_v1"},
        "summary": {
            "grade_distribution": df["risk_grade"].value_counts().to_dict(),
            "avg_pd": float(df["pd_score"].mean()),
        },
    }

    # Check both destinations before reserving files with exclusive creation.
    output = Path(output_dir)
    report_path = output / f"scoring_report_{run_id}.json"
    output_csv = output / f"scoring_output_{run_id}.csv"
    if report_path.exists() or output_csv.exists():
        raise FileExistsError("A scoring output already exists for this run ID")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8", newline="") as destination:
        df.to_csv(destination, index=False)
    with report_path.open("x", encoding="utf-8") as destination:
        destination.write(serialized)

    print(f"[OK] Report saved: {report_path}")
    print(f"[OK] Output saved: {output_csv}")

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Input CSV file path")
    parser.add_argument("--output", "-o", default="artifacts", help="Output directory")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, do not process"
    )

    args = parser.parse_args(argv)

    if args.validate_only:
        result = validate_csv(args.csv)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1

    try:
        run_scoring(args.csv, args.output)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
