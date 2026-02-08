"""E2E tests for run-real path"""
import subprocess
import json
import os
import glob

def test_run_real():
    """Test run-real path"""
    result = subprocess.run(
        ['python', 'scripts/run_real.py', 'data/sample_input.csv', '--output', 'artifacts'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"run-real failed: {result.stderr}"
    
    # Check output files exist
    report_files = glob.glob('artifacts/scoring_report_*.json')
    assert len(report_files) > 0, "No report file generated"
    
    # Validate JSON structure
    with open(report_files[0]) as f:
        report = json.load(f)
    
    assert 'run_id' in report
    assert 'version' in report
    assert 'timestamp' in report
    assert 'rows_processed' in report
