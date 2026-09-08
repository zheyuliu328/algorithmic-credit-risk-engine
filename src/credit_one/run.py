#!/usr/bin/env python3
"""CLI entry points for educational credit-risk demonstrations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run an offline synthetic classification experiment")
    demo.add_argument(
        "--seed", type=int, default=42, help="Simulation and split seed (default: 42)"
    )
    demo.add_argument(
        "--samples", type=int, default=1000, help="Synthetic sample count (minimum: 100)"
    )
    demo.add_argument("--output", "-o", default="artifacts/demo_report.json")

    validate = subparsers.add_parser("validate", help="Reserved; not implemented (exits nonzero)")
    validate.add_argument("--dry-run", action="store_true", help="Report unimplemented status only")
    validate.add_argument("--output", "-o", default="artifacts/validation_report.json")

    dashboard = subparsers.add_parser("dashboard", help="Launch the optional Streamlit interface")
    dashboard.add_argument("--port", "-p", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        if args.command == "demo":
            run_demo(args)
            return 0
        if args.command == "validate":
            return run_validate(args)
        return run_dashboard(args)
    except (ImportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def run_demo(args):
    """Fit on synthetic training rows and save recomputable held-out results."""
    if __package__:
        from .synthetic_demo import run_experiment
    else:
        from synthetic_demo import run_experiment

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    report = run_experiment(seed=args.seed, n_samples=args.samples)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(serialized)
    metrics = report["metrics"]
    print(f"Synthetic classification demo: seed={args.seed}, samples={args.samples}")
    print(
        f"Held-out AUC={metrics['auc']:.6f}, KS={metrics['ks']:.6f}, Brier={metrics['brier']:.6f}"
    )
    print(f"Report saved: {output}")
    print("Educational simulation; these are not calibrated credit-risk estimates.")
    return report


def run_validate(args):
    """Fail explicitly until a real validation orchestration path exists."""
    detail = " Dry run does not validate a model." if args.dry_run else ""
    print(
        f"NOT IMPLEMENTED: validate has no model-validation workflow.{detail} No report written.",
        file=sys.stderr,
    )
    return 2


def run_dashboard(args):
    """Use the current interpreter and propagate the Streamlit process status."""
    app = Path(__file__).resolve().with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
