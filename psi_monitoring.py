"""
PSI (Population Stability Index) Monitoring System
Production-grade model drift detection with three-tier alert system

Standard Thresholds:
- PSI < 0.1: Stable (Green)
- 0.1 ≤ PSI < 0.25: Warning (Yellow)  
- PSI ≥ 0.25: Critical (Red) - Investigation Required

Reference: SR 11-7 Model Risk Management Guidelines
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import sqlite3


class PSIMonitor:
    """
    Population Stability Index monitoring for credit risk models
    Tracks score distribution drift over time
    """
    
    # Industry-standard PSI thresholds
    THRESHOLDS = {
        'stable': 0.10,
        'warning': 0.25,
        'critical': float('inf')
    }
    
    def __init__(self, model_name: str, model_version: str, 
                 baseline_scores: np.ndarray = None,
                 db_path: str = 'psi_monitoring.db'):
        """
        Initialize PSI Monitor
        
        Parameters:
        -----------
        model_name : str
            Name of the model being monitored
        model_version : str
            Version of the model
        baseline_scores : np.ndarray
            Reference distribution (training/dev scores)
        db_path : str
            Path to SQLite database for logging
        """
        self.model_name = model_name
        self.model_version = model_version
        self.db_path = db_path
        
        if baseline_scores is not None:
            self.set_baseline(baseline_scores)
        else:
            self.baseline = None
            
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for PSI logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS psi_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                psi_score REAL NOT NULL,
                alert_level TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                period_start TEXT,
                period_end TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS psi_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER,
                bin_range TEXT,
                expected_pct REAL,
                actual_pct REAL,
                psi_contribution REAL,
                FOREIGN KEY (log_id) REFERENCES psi_logs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_baseline(self, scores: np.ndarray, save_path: str = 'baseline_distribution.json'):
        """
        Set baseline distribution from reference data
        
        Parameters:
        -----------
        scores : np.ndarray
            Reference score distribution (e.g., from training set)
        save_path : str
            Path to save baseline statistics
        """
        self.baseline = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentiles': np.percentile(scores, [10, 25, 50, 75, 90]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        baseline_stats = {k: v for k, v in self.baseline.items() if k != 'scores'}
        baseline_stats['percentiles'] = baseline_stats['percentiles'].tolist()
        
        with open(save_path, 'w') as f:
            json.dump(baseline_stats, f, indent=2)
        
        print(f"✓ Baseline set: n={len(scores)}, mean={self.baseline['mean']:.2f}")
    
    def calculate_psi(self, current_scores: np.ndarray, 
                     buckets: int = 10,
                     bucket_type: str = 'quantile') -> Dict:
        """
        Calculate Population Stability Index
        
        Parameters:
        -----------
        current_scores : np.ndarray
            Current period score distribution
        buckets : int
            Number of buckets for comparison
        bucket_type : str
            'quantile' (equal frequency) or 'fixed' (equal width)
            
        Returns:
        --------
        dict : PSI results with detailed breakdown
        """
        if self.baseline is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        baseline_scores = self.baseline['scores']
        
        # Create buckets
        if bucket_type == 'quantile':
            # Equal frequency buckets
            breakpoints = np.percentile(baseline_scores, 
                                       np.linspace(0, 100, buckets + 1))
        else:
            # Equal width buckets
            breakpoints = np.linspace(baseline_scores.min(), 
                                     baseline_scores.max(), 
                                     buckets + 1)
        
        breakpoints = np.unique(breakpoints)
        
        # Calculate percentages in each bucket
        def get_percentages(scores, breakpoints):
            counts, _ = np.histogram(scores, breakpoints)
            return counts / len(scores)
        
        expected_pct = get_percentages(baseline_scores, breakpoints)
        actual_pct = get_percentages(current_scores, breakpoints)
        
        # Calculate PSI for each bucket
        psi_details = []
        total_psi = 0
        
        for i in range(len(expected_pct)):
            bin_lower = breakpoints[i]
            bin_upper = breakpoints[i + 1] if i + 1 < len(breakpoints) else breakpoints[i]
            
            # Handle zero percentages (add small epsilon)
            e_pct = max(expected_pct[i], 0.0001)
            a_pct = max(actual_pct[i], 0.0001)
            
            # PSI formula: (Actual% - Expected%) × ln(Actual% / Expected%)
            psi_bin = (a_pct - e_pct) * np.log(a_pct / e_pct)
            total_psi += psi_bin
            
            psi_details.append({
                'bin_range': f'{bin_lower:.0f}-{bin_upper:.0f}',
                'expected_pct': round(e_pct * 100, 2),
                'actual_pct': round(a_pct * 100, 2),
                'psi_contribution': round(psi_bin, 6)
            })
        
        # Determine alert level
        if total_psi < self.THRESHOLDS['stable']:
            alert_level = 'STABLE'
            alert_color = '🟢'
        elif total_psi < self.THRESHOLDS['warning']:
            alert_level = 'WARNING'
            alert_color = '🟡'
        else:
            alert_level = 'CRITICAL'
            alert_color = '🔴'
        
        results = {
            'psi_score': round(total_psi, 4),
            'alert_level': alert_level,
            'alert_color': alert_color,
            'sample_size': len(current_scores),
            'baseline_size': len(baseline_scores),
            'timestamp': datetime.now().isoformat(),
            'thresholds': self.THRESHOLDS,
            'bucket_details': psi_details
        }
        
        return results
    
    def log_psi(self, psi_results: Dict, period_start: str = None, 
                period_end: str = None, notes: str = ''):
        """
        Log PSI results to database
        
        Parameters:
        -----------
        psi_results : dict
            Output from calculate_psi()
        period_start : str
            Start of monitoring period (ISO format)
        period_end : str
            End of monitoring period (ISO format)
        notes : str
            Additional notes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main log
        cursor.execute('''
            INSERT INTO psi_logs 
            (timestamp, model_name, model_version, psi_score, alert_level, 
             sample_size, period_start, period_end, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            psi_results['timestamp'],
            self.model_name,
            self.model_version,
            psi_results['psi_score'],
            psi_results['alert_level'],
            psi_results['sample_size'],
            period_start or psi_results['timestamp'],
            period_end or psi_results['timestamp'],
            notes
        ))
        
        log_id = cursor.lastrowid
        
        # Insert bucket details
        for bucket in psi_results['bucket_details']:
            cursor.execute('''
                INSERT INTO psi_details
                (log_id, bin_range, expected_pct, actual_pct, psi_contribution)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                log_id,
                bucket['bin_range'],
                bucket['expected_pct'],
                bucket['actual_pct'],
                bucket['psi_contribution']
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✓ PSI logged: {psi_results['alert_color']} {psi_results['alert_level']} "
              f"(PSI: {psi_results['psi_score']:.4f})")
    
    def get_psi_history(self, days: int = 30) -> pd.DataFrame:
        """
        Retrieve PSI monitoring history
        
        Parameters:
        -----------
        days : int
            Number of days to retrieve
            
        Returns:
        --------
        pd.DataFrame : Historical PSI data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, psi_score, alert_level, sample_size, notes
            FROM psi_logs
            WHERE model_name = ? AND model_version = ?
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, 
                              params=(self.model_name, self.model_version))
        conn.close()
        
        return df
    
    def generate_alert_report(self, psi_results: Dict) -> str:
        """
        Generate formatted alert report
        
        Parameters:
        -----------
        psi_results : dict
            Output from calculate_psi()
            
        Returns:
        --------
        str : Formatted report
        """
        report = f"""
{'='*60}
PSI MONITORING ALERT - {psi_results['alert_level']}
{'='*60}

Model: {self.model_name} v{self.model_version}
Timestamp: {psi_results['timestamp']}

PSI Score: {psi_results['psi_score']:.4f}
Alert Level: {psi_results['alert_color']} {psi_results['alert_level']}
Sample Size: {psi_results['sample_size']:,}

Thresholds:
  🟢 Stable:    PSI < {self.THRESHOLDS['stable']}
  🟡 Warning:   {self.THRESHOLDS['stable']} ≤ PSI < {self.THRESHOLDS['warning']}
  🔴 Critical:  PSI ≥ {self.THRESHOLDS['warning']}

Top Contributing Buckets:
"""
        
        # Sort by PSI contribution
        sorted_buckets = sorted(psi_results['bucket_details'], 
                               key=lambda x: abs(x['psi_contribution']), 
                               reverse=True)[:5]
        
        for bucket in sorted_buckets:
            report += f"  {bucket['bin_range']}: "
            report += f"E={bucket['expected_pct']:.1f}%, "
            report += f"A={bucket['actual_pct']:.1f}%, "
            report += f"PSI={bucket['psi_contribution']:.6f}\n"
        
        report += f"\n{'='*60}\n"
        
        if psi_results['alert_level'] == 'CRITICAL':
            report += """
⚠️  ACTION REQUIRED:
    - Investigate score distribution shift
    - Consider model retraining
    - Notify model risk management team
    - Document findings for audit
"""
        elif psi_results['alert_level'] == 'WARNING':
            report += """
⚠️  ATTENTION:
    - Monitor closely for continued drift
    - Analyze root cause of distribution shift
    - Prepare contingency plans
"""
        else:
            report += """
✓ Model is stable. Continue routine monitoring.
"""
        
        return report


def simulate_psi_monitoring():
    """
    Simulate PSI monitoring over time with various scenarios
    """
    np.random.seed(42)
    
    # Create monitor with baseline
    baseline_scores = np.random.normal(600, 50, 10000)
    monitor = PSIMonitor('XGBoost_PD_Model', 'v2.0', baseline_scores)
    
    print("="*60)
    print("PSI MONITORING SIMULATION")
    print("="*60)
    
    # Scenario 1: Stable period
    print("\n[Scenario 1: Stable Period]")
    current_scores = np.random.normal(600, 52, 5000)  # Slight increase in variance
    results = monitor.calculate_psi(current_scores)
    print(monitor.generate_alert_report(results))
    monitor.log_psi(results, notes="Weekly monitoring - stable")
    
    # Scenario 2: Warning - moderate drift
    print("\n[Scenario 2: Warning - Moderate Drift]")
    current_scores = np.random.normal(580, 55, 5000)  # Shift in mean
    results = monitor.calculate_psi(current_scores)
    print(monitor.generate_alert_report(results))
    monitor.log_psi(results, notes="Monthly monitoring - mean shift detected")
    
    # Scenario 3: Critical - significant drift
    print("\n[Scenario 3: Critical - Significant Drift]")
    current_scores = np.random.normal(550, 70, 5000)  # Large shift + more variance
    results = monitor.calculate_psi(current_scores)
    print(monitor.generate_alert_report(results))
    monitor.log_psi(results, notes="URGENT: Significant distribution shift")
    
    # Retrieve history
    print("\n[PSI History - Last 30 Days]")
    history = monitor.get_psi_history(days=30)
    print(history.to_string(index=False))


if __name__ == "__main__":
    simulate_psi_monitoring()
