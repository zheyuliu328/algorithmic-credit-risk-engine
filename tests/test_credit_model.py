"""Unit tests for credit risk model functionality."""

import pytest
import numpy as np
import pandas as pd
from sme_credit_explainability import (
    calculate_psi,
    monitor_model_stability,
    generate_synthetic_sme_data,
    train_scorecard_model,
    HAS_OPTBINNING
)


class TestPSICalculation:
    """Test PSI calculation functionality."""
    
    def test_psi_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi, _ = calculate_psi(data, data)
        assert psi < 0.01, "PSI should be near 0 for identical distributions"
    
    def test_psi_shifted_distribution(self):
        """PSI should be high for significantly shifted distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Mean shifted by 2
        psi, _ = calculate_psi(expected, actual)
        assert psi > 0.25, "PSI should be >0.25 for significantly different distributions"
    
    def test_psi_moderate_shift(self):
        """PSI should be moderate for slightly shifted distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)  # Small shift
        psi, _ = calculate_psi(expected, actual)
        assert 0.05 < psi < 0.3, "PSI should be moderate for slight shifts"
    
    def test_psi_insufficient_data(self):
        """PSI should handle insufficient unique values gracefully."""
        expected = np.array([1, 1, 1, 1, 1])
        actual = np.array([1, 1, 1, 1, 1])
        psi, details = calculate_psi(expected, actual)
        assert psi == 0.0
        assert 'warning' in details


class TestMonitoring:
    """Test model monitoring functionality."""
    
    def test_monitor_stability_output_format(self):
        """Monitor function should return properly formatted DataFrame."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'revenue_growth': np.random.normal(0.1, 0.05, 100),
            'debt_to_asset_ratio': np.random.uniform(0.2, 0.8, 100),
            'cash_flow_volatility': np.random.uniform(0.5, 2.0, 100)
        })
        
        current_data = train_data.copy()
        
        result = monitor_model_stability(train_data, current_data, 
                                        ['revenue_growth', 'debt_to_asset_ratio'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'Feature' in result.columns
        assert 'PSI' in result.columns
        assert 'Status' in result.columns
        assert 'Action' in result.columns
        assert len(result) == 2
    
    def test_monitor_stability_status_classification(self):
        """Monitor should correctly classify PSI status."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'revenue_growth': np.random.normal(0.1, 0.05, 100)
        })
        
        # Stable case
        current_stable = train_data.copy()
        result_stable = monitor_model_stability(train_data, current_stable, ['revenue_growth'])
        assert '🟢' in result_stable.iloc[0]['Status']
        
        # Drifted case
        current_drift = pd.DataFrame({
            'revenue_growth': np.random.normal(0.5, 0.05, 100)
        })
        result_drift = monitor_model_stability(train_data, current_drift, ['revenue_growth'])
        assert '🔴' in result_drift.iloc[0]['Status'] or '🟡' in result_drift.iloc[0]['Status']


@pytest.mark.skipif(not HAS_OPTBINNING, reason="optbinning not installed")
class TestCreditScoring:
    """Test credit scoring logic."""
    
    def test_scorecard_training(self):
        """Scorecard should train successfully."""
        df = generate_synthetic_sme_data(n_samples=500)
        scorecard, test_df = train_scorecard_model(df)
        
        assert scorecard.is_fitted
        assert len(test_df) > 0
        assert 'predicted_default_prob' in test_df.columns
    
    def test_score_range(self):
        """Credit scores should be within 300-850 range."""
        df = generate_synthetic_sme_data(n_samples=500)
        scorecard, test_df = train_scorecard_model(df)
        
        scores = scorecard.predict_score(test_df[['revenue_growth', 'debt_to_asset_ratio', 
                                                   'cash_flow_volatility', 'industry', 'past_default']])
        
        assert all(300 <= s <= 850 for s in scores), "All scores should be in 300-850 range"
    
    def test_high_debt_lowers_score(self):
        """Higher debt-to-asset ratio should result in lower credit score."""
        df = generate_synthetic_sme_data(n_samples=500)
        scorecard, _ = train_scorecard_model(df)
        
        # Create two identical companies except debt ratio
        low_debt = pd.DataFrame([{
            'revenue_growth': 0.1,
            'debt_to_asset_ratio': 0.2,  # Low debt
            'cash_flow_volatility': 1.0,
            'industry': 'Tech',
            'past_default': 0
        }])
        
        high_debt = pd.DataFrame([{
            'revenue_growth': 0.1,
            'debt_to_asset_ratio': 0.9,  # High debt
            'cash_flow_volatility': 1.0,
            'industry': 'Tech',
            'past_default': 0
        }])
        
        score_low = scorecard.predict_score(low_debt)[0]
        score_high = scorecard.predict_score(high_debt)[0]
        
        assert score_low > score_high, "Lower debt should result in higher credit score"


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_synthetic_data_shape(self):
        """Generated data should have correct shape and columns."""
        df = generate_synthetic_sme_data(n_samples=100)
        
        assert len(df) == 100
        assert 'revenue_growth' in df.columns
        assert 'debt_to_asset_ratio' in df.columns
        assert 'cash_flow_volatility' in df.columns
        assert 'industry' in df.columns
        assert 'past_default' in df.columns
        assert 'true_label' in df.columns
        assert 'company_id' in df.columns
    
    def test_vip_clients_quality(self):
        """First 6 VIP clients should have favorable metrics."""
        df = generate_synthetic_sme_data()
        
        vip_df = df.head(6)
        
        # VIPs should have no past defaults
        assert all(vip_df['past_default'] == 0)
        
        # VIPs should have positive revenue growth
        assert all(vip_df['revenue_growth'] > 0)
        
        # VIPs should have reasonable debt ratios
        assert all(vip_df['debt_to_asset_ratio'] < 0.5)


# Run tests with: pytest tests/test_credit_model.py -v
