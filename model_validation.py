"""
Model Validation Framework - Production-grade model validation
Implements SR 11-7 compliant model risk management

Key Components:
- Out-of-Time (OOT) Validation
- K-S Test, CAP Curve, Gini Coefficient
- Model Stability Analysis
- Calibration Assessment
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sqlite3


class ModelValidator:
    """
    Comprehensive model validation framework for credit risk models
    """
    
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.validation_results = {}
        self.timestamp = datetime.now().isoformat()
        
    def out_of_time_validation(self, y_true_train, y_pred_train, y_true_oot, y_pred_oot,
                                train_period, oot_period):
        """
        Out-of-Time validation to detect temporal overfitting
        
        Parameters:
        -----------
        y_true_train : array-like
            True labels for training period
        y_pred_train : array-like
            Predicted probabilities for training period
        y_true_oot : array-like
            True labels for OOT period
        y_pred_oot : array-like
            Predicted probabilities for OOT period
        train_period : str
            Description of training period (e.g., "2022-01 to 2022-12")
        oot_period : str
            Description of OOT period (e.g., "2023-01 to 2023-06")
            
        Returns:
        --------
        dict : OOT validation metrics
        """
        # Calculate AUC for both periods
        auc_train = roc_auc_score(y_true_train, y_pred_train)
        auc_oot = roc_auc_score(y_true_oot, y_pred_oot)
        
        # AUC degradation check (should be < 0.05)
        auc_degradation = auc_train - auc_oot
        
        # Population stability
        psi_score = self._calculate_psi(y_pred_train, y_pred_oot)
        
        results = {
            'train_auc': round(auc_train, 4),
            'oot_auc': round(auc_oot, 4),
            'auc_degradation': round(auc_degradation, 4),
            'degradation_acceptable': auc_degradation < 0.05,
            'psi_score': round(psi_score, 4),
            'psi_acceptable': psi_score < 0.25,
            'train_period': train_period,
            'oot_period': oot_period,
            'train_samples': len(y_true_train),
            'oot_samples': len(y_true_oot)
        }
        
        self.validation_results['oot_validation'] = results
        return results
    
    def ks_test(self, y_true, y_pred):
        """
        Kolmogorov-Smirnov test for model discrimination power
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted probabilities
            
        Returns:
        --------
        dict : K-S test results
        """
        # Split predictions by actual outcome
        pred_good = y_pred[y_true == 0]
        pred_bad = y_pred[y_true == 1]
        
        # Perform K-S test
        ks_statistic, p_value = stats.ks_2samp(pred_good, pred_bad)
        
        # Calculate K-S score (maximum distance between CDFs)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks_score = max(tpr - fpr)
        
        results = {
            'ks_statistic': round(ks_statistic, 4),
            'ks_score': round(ks_score, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'discrimination_power': 'Strong' if ks_score > 0.4 else 'Moderate' if ks_score > 0.3 else 'Weak'
        }
        
        self.validation_results['ks_test'] = results
        return results
    
    def cap_curve_analysis(self, y_true, y_pred):
        """
        Cumulative Accuracy Profile (CAP) Curve and Accuracy Ratio
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted probabilities
            
        Returns:
        --------
        dict : CAP curve metrics
        """
        n_total = len(y_true)
        n_bad = sum(y_true)
        
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate cumulative bads
        cumsum_bad = np.cumsum(y_true_sorted)
        cap_curve = cumsum_bad / n_bad
        
        # Random model (diagonal)
        x_axis = np.arange(1, n_total + 1) / n_total
        
        # Perfect model
        perfect_curve = np.minimum(x_axis * n_total / n_bad, 1)
        
        # Calculate Accuracy Ratio (Gini coefficient)
        ar_random = 0.5
        ar_model = np.trapz(cap_curve, x_axis)
        ar_perfect = np.trapz(perfect_curve, x_axis)
        
        accuracy_ratio = (ar_model - ar_random) / (ar_perfect - ar_random)
        gini_coefficient = 2 * accuracy_ratio
        
        results = {
            'accuracy_ratio': round(accuracy_ratio, 4),
            'gini_coefficient': round(gini_coefficient, 4),
            'model_auc': round(ar_model, 4),
            'interpretation': 'Excellent' if accuracy_ratio > 0.8 else 'Good' if accuracy_ratio > 0.6 else 'Acceptable' if accuracy_ratio > 0.4 else 'Poor'
        }
        
        self.validation_results['cap_curve'] = results
        return results
    
    def calibration_assessment(self, y_true, y_pred, n_bins=10):
        """
        Assess probability calibration using binning approach
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted probabilities
        n_bins : int
            Number of bins for calibration
            
        Returns:
        --------
        dict : Calibration metrics
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                calibration_data.append({
                    'bin_range': f'{bin_lower:.1f}-{bin_upper:.1f}',
                    'predicted_rate': round(avg_confidence_in_bin, 4),
                    'actual_rate': round(accuracy_in_bin, 4),
                    'samples': int(in_bin.sum())
                })
        
        # Calculate Expected Calibration Error (ECE)
        ece = sum([abs(d['predicted_rate'] - d['actual_rate']) * d['samples'] 
                   for d in calibration_data]) / len(y_pred)
        
        results = {
            'expected_calibration_error': round(ece, 4),
            'calibration_acceptable': ece < 0.05,
            'bin_details': calibration_data
        }
        
        self.validation_results['calibration'] = results
        return results
    
    def generate_validation_report(self, output_path='model_validation_report.json'):
        """
        Generate comprehensive validation report
        
        Parameters:
        -----------
        output_path : str
            Path to save validation report
        """
        report = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'validation_timestamp': self.timestamp,
            'validation_framework': 'SR 11-7 Compliant',
            'results': self.validation_results,
            'overall_assessment': self._assess_overall()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Validation report saved to {output_path}")
        return report
    
    def _calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index"""
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input
        
        breakpoints = np.linspace(0, 1, buckets + 1)
        breakpoints = np.percentile(expected, breakpoints * 100)
        breakpoints = np.unique(breakpoints)
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            return (e_perc - a_perc) * np.log(e_perc / a_perc)
        
        psi_value = sum([sub_psi(expected_percents[i], actual_percents[i]) 
                        for i in range(len(expected_percents))])
        
        return psi_value
    
    def _assess_overall(self):
        """Generate overall model assessment"""
        assessments = []
        
        if 'oot_validation' in self.validation_results:
            oot = self.validation_results['oot_validation']
            assessments.append({
                'dimension': 'Out-of-Time Stability',
                'status': 'PASS' if oot['degradation_acceptable'] and oot['psi_acceptable'] else 'FAIL',
                'details': f"AUC degradation: {oot['auc_degradation']:.4f}, PSI: {oot['psi_score']:.4f}"
            })
        
        if 'ks_test' in self.validation_results:
            ks = self.validation_results['ks_test']
            assessments.append({
                'dimension': 'Discrimination Power (K-S)',
                'status': 'PASS' if ks['ks_score'] > 0.3 else 'WARNING',
                'details': f"K-S score: {ks['ks_score']:.4f}, Power: {ks['discrimination_power']}"
            })
        
        if 'cap_curve' in self.validation_results:
            cap = self.validation_results['cap_curve']
            assessments.append({
                'dimension': 'Accuracy Ratio (Gini)',
                'status': 'PASS' if cap['accuracy_ratio'] > 0.4 else 'WARNING',
                'details': f"AR: {cap['accuracy_ratio']:.4f}, Gini: {cap['gini_coefficient']:.4f}"
            })
        
        if 'calibration' in self.validation_results:
            cal = self.validation_results['calibration']
            assessments.append({
                'dimension': 'Probability Calibration',
                'status': 'PASS' if cal['calibration_acceptable'] else 'WARNING',
                'details': f"ECE: {cal['expected_calibration_error']:.4f}"
            })
        
        return assessments


def run_full_validation(y_true_train, y_pred_train, y_true_test, y_pred_test,
                        y_true_oot=None, y_pred_oot=None,
                        model_name='CreditRiskModel', model_version='v1.0'):
    """
    Run complete validation suite
    
    Example usage:
    --------------
    results = run_full_validation(
        y_true_train=train_labels,
        y_pred_train=train_preds,
        y_true_test=test_labels,
        y_pred_test=test_preds,
        y_true_oot=oot_labels,  # Out-of-time validation set
        y_pred_oot=oot_preds,
        model_name='XGBoost_PD_Model',
        model_version='v2.1'
    )
    """
    validator = ModelValidator(model_name, model_version)
    
    print("="*60)
    print(f"MODEL VALIDATION REPORT: {model_name} {model_version}")
    print("="*60)
    
    # 1. OOT Validation (if provided)
    if y_true_oot is not None and y_pred_oot is not None:
        print("\n[1/4] Out-of-Time Validation")
        oot_results = validator.out_of_time_validation(
            y_true_train, y_pred_train, y_true_oot, y_pred_oot,
            train_period="Training Period",
            oot_period="Out-of-Time Period"
        )
        print(f"  Train AUC: {oot_results['train_auc']:.4f}")
        print(f"  OOT AUC: {oot_results['oot_auc']:.4f}")
        print(f"  Degradation: {oot_results['auc_degradation']:.4f} {'✓' if oot_results['degradation_acceptable'] else '✗'}")
        print(f"  PSI: {oot_results['psi_score']:.4f} {'✓' if oot_results['psi_acceptable'] else '✗'}")
    
    # 2. K-S Test
    print("\n[2/4] Kolmogorov-Smirnov Test")
    ks_results = validator.ks_test(y_true_test, y_pred_test)
    print(f"  K-S Score: {ks_results['ks_score']:.4f}")
    print(f"  P-value: {ks_results['p_value']:.6f}")
    print(f"  Power: {ks_results['discrimination_power']}")
    
    # 3. CAP Curve
    print("\n[3/4] CAP Curve Analysis")
    cap_results = validator.cap_curve_analysis(y_true_test, y_pred_test)
    print(f"  Accuracy Ratio: {cap_results['accuracy_ratio']:.4f}")
    print(f"  Gini Coefficient: {cap_results['gini_coefficient']:.4f}")
    print(f"  Interpretation: {cap_results['interpretation']}")
    
    # 4. Calibration
    print("\n[4/4] Calibration Assessment")
    cal_results = validator.calibration_assessment(y_true_test, y_pred_test)
    print(f"  Expected Calibration Error: {cal_results['expected_calibration_error']:.4f}")
    print(f"  Status: {'✓ Acceptable' if cal_results['calibration_acceptable'] else '⚠ Review needed'}")
    
    # Generate report
    report = validator.generate_validation_report()
    
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    for assessment in report['overall_assessment']:
        status_icon = "✓" if assessment['status'] == 'PASS' else "⚠" if assessment['status'] == 'WARNING' else "✗"
        print(f"{status_icon} {assessment['dimension']}: {assessment['status']}")
        print(f"   {assessment['details']}")
    
    return report


if __name__ == "__main__":
    # Example: Generate synthetic validation data
    np.random.seed(42)
    
    # Simulate model predictions
    n_train, n_test, n_oot = 8000, 2000, 2000
    
    # Training data
    y_train = np.random.binomial(1, 0.15, n_train)
    y_pred_train = np.clip(y_train * 0.7 + np.random.beta(2, 5, n_train) * 0.3, 0, 1)
    
    # Test data
    y_test = np.random.binomial(1, 0.15, n_test)
    y_pred_test = np.clip(y_test * 0.7 + np.random.beta(2, 5, n_test) * 0.3, 0, 1)
    
    # OOT data (slightly different distribution to simulate time decay)
    y_oot = np.random.binomial(1, 0.18, n_oot)
    y_pred_oot = np.clip(y_oot * 0.65 + np.random.beta(2, 4, n_oot) * 0.35, 0, 1)
    
    # Run validation
    results = run_full_validation(
        y_train, y_pred_train,
        y_test, y_pred_test,
        y_oot, y_pred_oot,
        model_name='XGBoost_PD_Model',
        model_version='v2.0_PRODUCTION'
    )
