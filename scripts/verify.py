"""Run supported repository checks in a new directory without deleting artifacts."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_checked(command, *, cwd, env):
    """Preserve a failed check's process status; never convert it into a warning."""
    print("Checking:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def new_output_directory(requested=None):
    """Reserve a new destination, leaving every existing path untouched."""
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="credit-risk-verify-"))
    output = Path(requested).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def verify(output):
    """Run lint, full tests, configuration, fresh demos and wheel packaging."""
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(output / "matplotlib"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    for path in ["src/credit_one/run.py", "pyproject.toml", "LICENSE", ".env.example"]:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"Required repository file missing: {path}")

    commands = [
        [sys.executable, "-m", "ruff", "check", ".", "--no-cache"],
        [sys.executable, "-m", "black", "--check", "."],
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        [sys.executable, "config/validator.py", "config/config.yaml"],
        [
            sys.executable,
            "src/credit_one/run.py",
            "demo",
            "--seed",
            "42",
            "--output",
            str(output / "demo_report.json"),
        ],
        [
            sys.executable,
            "scripts/run_real.py",
            "data/sample_input.csv",
            "--output",
            str(output / "csv"),
        ],
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output / "wheels"),
        ],
    ]
    for command in commands:
        run_checked(command, cwd=ROOT, env=env)

    demo = json.loads((output / "demo_report.json").read_text())
    if demo["mode"] != "SYNTHETIC_CLASSIFICATION_DEMO" or not demo["held_out"]:
        raise ValueError("Fresh synthetic evaluation evidence is missing")
    reports = list((output / "csv").glob("scoring_report_*.json"))
    if len(reports) != 1 or json.loads(reports[0].read_text())["rows_processed"] != 3:
        raise ValueError("Fresh CSV demonstration evidence is missing")
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one freshly built wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        required = {"credit_one/__init__.py", "credit_one/run.py", "credit_one/synthetic_demo.py"}
        if not required.issubset(wheel.namelist()):
            raise ValueError("Built wheel is missing the supported Python package")

    summary = {
        "status": "SUPPORTED_CHECKS_COMPLETED",
        "checks": len(commands),
        "output_directory": str(output),
        "scope": "Installation, code checks, current complete tests and offline examples only",
        "limitations": [
            "No regulatory approval, calibration or production readiness is established.",
            "Standalone validation and ECL integration remain unimplemented.",
            "The independent Security workflow performs the Git history gitleaks scan.",
            "Optional online integrations and the dashboard are outside these offline checks.",
        ],
    }
    with (output / "verification.json").open("x", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")
    print(f"Supported checks completed. New evidence: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="A new directory; an existing path is rejected")
    args = parser.parse_args(argv)
    try:
        output = new_output_directory(args.output_dir)
        print(f"Verification evidence: {output}", flush=True)
        verify(output)
    except subprocess.CalledProcessError as exc:
        print(f"CHECK FAILED with status {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
