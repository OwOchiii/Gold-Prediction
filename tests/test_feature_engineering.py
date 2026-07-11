"""
Unit tests for Feature Engineering Module

Tests cover:
- Lag feature creation
- Rolling statistics features
- Technical indicators (RSI, MACD, Bollinger Bands)
- Interaction features
- Temporal features
- Complete feature set building
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import FeatureEngineer, engineer_features
from config import Config


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data with trend
    base_price = 1500
    trend = np.linspace(0, 200, len(dates))
    noise = np.random.randn(len(dates)) * 10
    close_prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(len(dates)) * 5,
        'High': close_prices + np.abs(np.random.randn(len(dates)) * 10),
        'Low': close_prices - np.abs(np.random.randn(len(dates)) * 10),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    return df


@pytest.fixture
def sample_data_with_indicators():
    """Create sample data with economic indicators."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    base_price = 1500
    trend = np.linspace(0, 200, len(dates))
    noise = np.random.randn(len(dates)) * 10
    close_prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(len(dates)) * 5,
        'High': close_prices + np.abs(np.random.randn(len(dates)) * 10),
        'Low': close_prices - np.abs(np.random.randn(len(dates)) * 10),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, len(dates)),
        'DXY': 90 + np.random.randn(len(dates)) * 2,
        'Oil': 50 + np.random.randn(len(dates)) * 5,
        'Treasury_10Y': 1.5 + np.random.randn(len(dates)) * 0.2
    }, index=dates)
    
    return df


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


class TestLagFeatures:
    """Tests for lag feature creation."""
    
    def test_create_lag_features_default(self, feature_engineer, sample_ohlcv_data):
        """Test creating lag features with default parameters."""
        df_with_lags = feature_engineer.create_lag_features(sample_ohlcv_data)
        
        # Check that lag columns were created
        for lag in Config.LAG_PERIODS:
            assert f'Close_lag_{lag}' in df_with_lags.columns
        
        # Check that lag values are correct
        assert df_with_lags['Close_lag_1'].iloc[1] == sample_ohlcv_data['Close'].iloc[0]
        assert df_with_lags['Close_lag_7'].iloc[7] == sample_ohlcv_data['Close'].iloc[0]
    
    def test_create_lag_features_custom_lags(self, feature_engineer, sample_ohlcv_data):
        """Test creating lag features with custom lag periods."""
        custom_lags = [1, 5, 10]
        df_with_lags = feature_engineer.create_lag_features(sample_ohlcv_data, lags=custom_lags)
        
        # Check that custom lag columns were created
        for lag in custom_lags:
            assert f'Close_lag_{lag}' in df_with_lags.columns
        
        # Check that default lags not present if not specified
        assert 'Close_lag_14' not in df_with_lags.columns
    
    def test_create_lag_features_custom_column(self, feature_engineer, sample_ohlcv_data):
        """Test creating lag features for custom column."""
        df_with_lags = feature_engineer.create_lag_features(sample_ohlcv_data, 
                                                            column='Open', 
                                                            lags=[1, 7])
        
        assert 'Open_lag_1' in df_with_lags.columns
        assert 'Open_lag_7' in df_with_lags.columns
    
    def test_lag_features_nan_values(self, feature_engineer, sample_ohlcv_data):
        """Test that lag features introduce expected NaN values."""
        df_with_lags = feature_engineer.create_lag_features(sample_ohlcv_data, lags=[1, 7, 30])
        
        # First lag period should have NaN
        assert df_with_lags['Close_lag_1'].iloc[0] is pd.NaT or pd.isna(df_with_lags['Close_lag_1'].iloc[0])
        
        # 30th lag should have 30 NaN values
        assert df_with_lags['Close_lag_30'].isna().sum() == 30


class TestRollingFeatures:
    """Tests for rolling statistics features."""
    
    def test_create_rolling_features_mean(self, feature_engineer, sample_ohlcv_data):
        """Test creating rolling mean features."""
        df_with_rolling = feature_engineer.create_rolling_features(sample_ohlcv_data)
        
        # Check that rolling mean columns were created
        for window in Config.ROLLING_WINDOWS:
            assert f'Close_ma_{window}' in df_with_rolling.columns
    
    def test_create_rolling_features_std(self, feature_engineer, sample_ohlcv_data):
        """Test creating rolling standard deviation features."""
        df_with_rolling = feature_engineer.create_rolling_features(sample_ohlcv_data)
        
        # Check that rolling std columns were created
        for window in Config.ROLLING_STD_WINDOWS:
            assert f'Close_std_{window}' in df_with_rolling.columns
    
    def test_rolling_mean_calculation(self, feature_engineer, sample_ohlcv_data):
        """Test that rolling mean is calculated correctly."""
        df_with_rolling = feature_engineer.create_rolling_features(sample_ohlcv_data, 
                                                                   windows=[7])
        
        # Calculate expected 7-day moving average at position 10
        expected_ma = sample_ohlcv_data['Close'].iloc[4:11].mean()
        actual_ma = df_with_rolling['Close_ma_7'].iloc[10]
        
        assert np.isclose(expected_ma, actual_ma, rtol=1e-5)
    
    def test_rolling_std_calculation(self, feature_engineer, sample_ohlcv_data):
        """Test that rolling std is calculated correctly."""
        df_with_rolling = feature_engineer.create_rolling_features(sample_ohlcv_data,
                                                                   windows=[],
                                                                   std_windows=[7])
        
        # Calculate expected 7-day std at position 10
        expected_std = sample_ohlcv_data['Close'].iloc[4:11].std()
        actual_std = df_with_rolling['Close_std_7'].iloc[10]
        
        assert np.isclose(expected_std, actual_std, rtol=1e-5)
    
    def test_rolling_features_custom_column(self, feature_engineer, sample_ohlcv_data):
        """Test creating rolling features for custom column."""
        df_with_rolling = feature_engineer.create_rolling_features(sample_ohlcv_data,
                                                                   column='Volume',
                                                                   windows=[7, 14],
                                                                   std_windows=[7])
        
        assert 'Volume_ma_7' in df_with_rolling.columns
        assert 'Volume_ma_14' in df_with_rolling.columns
        assert 'Volume_std_7' in df_with_rolling.columns


class TestTechnicalIndicators:
    """Tests for technical indicator creation."""
    
    def test_create_technical_indicators_columns(self, feature_engineer, sample_ohlcv_data):
        """Test that all technical indicator columns are created."""
        df_with_indicators = feature_engineer.create_technical_indicators(sample_ohlcv_data)
        
        # Check RSI
        assert f'RSI_{Config.RSI_PERIOD}' in df_with_indicators.columns
        
        # Check MACD
        assert 'MACD' in df_with_indicators.columns
        assert 'MACD_signal' in df_with_indicators.columns
        assert 'MACD_diff' in df_with_indicators.columns
        
        # Check Bollinger Bands
        assert 'BB_upper' in df_with_indicators.columns
        assert 'BB_middle' in df_with_indicators.columns
        assert 'BB_lower' in df_with_indicators.columns
    
    def test_rsi_range(self, feature_engineer, sample_ohlcv_data):
        """Test that RSI values are in valid range [0, 100]."""
        df_with_indicators = feature_engineer.create_technical_indicators(sample_ohlcv_data)
        
        rsi_col = f'RSI_{Config.RSI_PERIOD}'
        rsi_values = df_with_indicators[rsi_col].dropna()
        
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100
    
    def test_bollinger_bands_relationship(self, feature_engineer, sample_ohlcv_data):
        """Test that Bollinger Bands maintain correct relationship: lower < middle < upper."""
        df_with_indicators = feature_engineer.create_technical_indicators(sample_ohlcv_data)
        
        # Remove NaN values
        df_clean = df_with_indicators[['BB_lower', 'BB_middle', 'BB_upper']].dropna()
        
        # Check relationships
        assert (df_clean['BB_lower'] <= df_clean['BB_middle']).all()
        assert (df_clean['BB_middle'] <= df_clean['BB_upper']).all()
    
    def test_macd_difference_calculation(self, feature_engineer, sample_ohlcv_data):
        """Test that MACD difference equals MACD - MACD_signal."""
        df_with_indicators = feature_engineer.create_technical_indicators(sample_ohlcv_data)
        
        df_clean = df_with_indicators[['MACD', 'MACD_signal', 'MACD_diff']].dropna()
        
        expected_diff = df_clean['MACD'] - df_clean['MACD_signal']
        actual_diff = df_clean['MACD_diff']
        
        assert np.allclose(expected_diff, actual_diff, rtol=1e-5)
    
    def test_technical_indicators_without_close_column(self, feature_engineer):
        """Test that error is raised when Close column is missing."""
        df_no_close = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        with pytest.raises(ValueError, match="Close column required"):
            feature_engineer.create_technical_indicators(df_no_close)


class TestInteractionFeatures:
    """Tests for interaction feature creation."""
    
    def test_gold_oil_ratio(self, feature_engineer, sample_data_with_indicators):
        """Test Gold/Oil ratio calculation."""
        df_with_interactions = feature_engineer.create_interaction_features(sample_data_with_indicators)
        
        assert 'Gold_Oil_ratio' in df_with_interactions.columns
        
        # Verify calculation
        expected_ratio = sample_data_with_indicators['Close'] / sample_data_with_indicators['Oil']
        actual_ratio = df_with_interactions['Gold_Oil_ratio']
        
        assert np.allclose(expected_ratio, actual_ratio, rtol=1e-5)
    
    def test_gold_dxy_correlation(self, feature_engineer, sample_data_with_indicators):
        """Test Gold/DXY correlation feature."""
        df_with_interactions = feature_engineer.create_interaction_features(sample_data_with_indicators)
        
        assert 'Gold_DXY_corr' in df_with_interactions.columns
        
        # Check that correlation values are in valid range [-1, 1]
        corr_values = df_with_interactions['Gold_DXY_corr'].dropna()
        assert corr_values.min() >= -1
        assert corr_values.max() <= 1
    
    def test_gold_treasury_spread(self, feature_engineer, sample_data_with_indicators):
        """Test Gold/Treasury spread calculation."""
        df_with_interactions = feature_engineer.create_interaction_features(sample_data_with_indicators)
        
        assert 'Gold_Treasury_spread' in df_with_interactions.columns
    
    def test_interaction_features_missing_indicators(self, feature_engineer, sample_ohlcv_data):
        """Test interaction features when economic indicators are missing."""
        # Should not raise error, just skip features
        df_with_interactions = feature_engineer.create_interaction_features(sample_ohlcv_data)
        
        # Interaction features should not be present
        assert 'Gold_Oil_ratio' not in df_with_interactions.columns
        assert 'Gold_DXY_corr' not in df_with_interactions.columns


class TestTemporalFeatures:
    """Tests for temporal feature creation."""
    
    def test_create_temporal_features_columns(self, feature_engineer, sample_ohlcv_data):
        """Test that all temporal feature columns are created."""
        df_with_temporal = feature_engineer.create_temporal_features(sample_ohlcv_data)
        
        expected_features = ['day_of_week', 'month', 'quarter', 'year', 
                           'is_quarter_end', 'is_year_end']
        
        for feature in expected_features:
            assert feature in df_with_temporal.columns
    
    def test_day_of_week_range(self, feature_engineer, sample_ohlcv_data):
        """Test that day_of_week is in valid range [0, 6]."""
        df_with_temporal = feature_engineer.create_temporal_features(sample_ohlcv_data)
        
        assert df_with_temporal['day_of_week'].min() >= 0
        assert df_with_temporal['day_of_week'].max() <= 6
    
    def test_month_range(self, feature_engineer, sample_ohlcv_data):
        """Test that month is in valid range [1, 12]."""
        df_with_temporal = feature_engineer.create_temporal_features(sample_ohlcv_data)
        
        assert df_with_temporal['month'].min() >= 1
        assert df_with_temporal['month'].max() <= 12
    
    def test_quarter_range(self, feature_engineer, sample_ohlcv_data):
        """Test that quarter is in valid range [1, 4]."""
        df_with_temporal = feature_engineer.create_temporal_features(sample_ohlcv_data)
        
        assert df_with_temporal['quarter'].min() >= 1
        assert df_with_temporal['quarter'].max() <= 4
    
    def test_year_values(self, feature_engineer, sample_ohlcv_data):
        """Test that year values are correct."""
        df_with_temporal = feature_engineer.create_temporal_features(sample_ohlcv_data)
        
        # Sample data is from 2020
        assert (df_with_temporal['year'] == 2020).all()
    
    def test_is_quarter_end_values(self, feature_engineer):
        """Test that is_quarter_end correctly identifies quarter ends."""
        # Create data with known quarter ends
        dates = pd.DatetimeIndex(['2020-03-31', '2020-04-01', '2020-06-30', '2020-09-30', '2020-12-31'])
        df = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        df_with_temporal = feature_engineer.create_temporal_features(df)
        
        # Check quarter end flags
        assert df_with_temporal['is_quarter_end'].iloc[0] == 1  # March 31
        assert df_with_temporal['is_quarter_end'].iloc[1] == 0  # April 1
        assert df_with_temporal['is_quarter_end'].iloc[2] == 1  # June 30
    
    def test_is_year_end_values(self, feature_engineer):
        """Test that is_year_end correctly identifies year ends."""
        dates = pd.DatetimeIndex(['2020-12-30', '2020-12-31', '2021-01-01'])
        df = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
        
        df_with_temporal = feature_engineer.create_temporal_features(df)
        
        assert df_with_temporal['is_year_end'].iloc[0] == 0  # Dec 30
        assert df_with_temporal['is_year_end'].iloc[1] == 1  # Dec 31
        assert df_with_temporal['is_year_end'].iloc[2] == 0  # Jan 1
    
    def test_temporal_features_without_datetime_index(self, feature_engineer):
        """Test that error is raised when DataFrame doesn't have DatetimeIndex."""
        df_no_datetime = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        with pytest.raises(ValueError, match="DatetimeIndex"):
            feature_engineer.create_temporal_features(df_no_datetime)


class TestCompleteFeatureSet:
    """Tests for complete feature set building."""
    
    def test_build_feature_set_all_features(self, feature_engineer, sample_data_with_indicators):
        """Test building complete feature set with all features enabled."""
        df_complete = feature_engineer.build_feature_set(sample_data_with_indicators)
        
        # Check that original columns are preserved
        assert 'Close' in df_complete.columns
        
        # Check that lag features exist
        assert 'Close_lag_1' in df_complete.columns
        
        # Check that rolling features exist
        assert 'Close_ma_7' in df_complete.columns
        assert 'Close_std_7' in df_complete.columns
        
        # Check that technical indicators exist
        assert f'RSI_{Config.RSI_PERIOD}' in df_complete.columns
        assert 'MACD' in df_complete.columns
        assert 'BB_upper' in df_complete.columns
        
        # Check that interaction features exist
        assert 'Gold_Oil_ratio' in df_complete.columns
        
        # Check that temporal features exist
        assert 'day_of_week' in df_complete.columns
    
    def test_build_feature_set_selective(self, feature_engineer, sample_data_with_indicators):
        """Test building feature set with selective features."""
        df_selective = feature_engineer.build_feature_set(
            sample_data_with_indicators,
            create_lags=True,
            create_rolling=False,
            create_technical=False,
            create_interactions=False,
            create_temporal=True,
            handle_nan=False
        )
        
        # Lag features should exist
        assert 'Close_lag_1' in df_selective.columns
        
        # Rolling features should not exist
        assert 'Close_ma_7' not in df_selective.columns
        
        # Technical indicators should not exist
        assert 'MACD' not in df_selective.columns
        
        # Temporal features should exist
        assert 'day_of_week' in df_selective.columns
    
    def test_build_feature_set_nan_handling(self, feature_engineer, sample_data_with_indicators):
        """Test that NaN handling drops rows with missing values."""
        df_with_nan = feature_engineer.build_feature_set(
            sample_data_with_indicators,
            handle_nan=False
        )
        
        df_without_nan = feature_engineer.build_feature_set(
            sample_data_with_indicators,
            handle_nan=True
        )
        
        # With NaN handling, no NaN values should remain
        assert df_without_nan.isna().sum().sum() == 0
        
        # Without NaN handling, some NaN values should remain
        assert df_with_nan.isna().sum().sum() > 0
        
        # With NaN handling, fewer rows should remain
        assert len(df_without_nan) < len(df_with_nan)
    
    def test_build_feature_set_shape(self, feature_engineer, sample_data_with_indicators):
        """Test that feature set has expected number of columns."""
        initial_cols = len(sample_data_with_indicators.columns)
        
        df_complete = feature_engineer.build_feature_set(sample_data_with_indicators)
        
        final_cols = len(df_complete.columns)
        
        # Many features should have been added
        assert final_cols > initial_cols + 20


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_engineer_features_function(self, sample_data_with_indicators):
        """Test the convenience engineer_features function."""
        df_engineered = engineer_features(sample_data_with_indicators)
        
        # Should have lag features
        assert 'Close_lag_1' in df_engineered.columns
        
        # Should have rolling features
        assert 'Close_ma_7' in df_engineered.columns
        
        # Should have technical indicators
        assert 'MACD' in df_engineered.columns
        
        # Should have temporal features
        assert 'day_of_week' in df_engineered.columns
        
        # Should have no NaN values (default behavior)
        assert df_engineered.isna().sum().sum() == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_dataframe(self, feature_engineer):
        """Test handling of empty DataFrame."""
        df_empty = pd.DataFrame()
        
        # Should handle gracefully (may raise or return empty)
        # Implementation specific behavior
        pass
    
    def test_small_dataframe(self, feature_engineer):
        """Test handling of DataFrame smaller than window sizes."""
        dates = pd.date_range('2020-01-01', periods=5)
        df_small = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        # Should create features even if many are NaN
        df_with_features = feature_engineer.create_rolling_features(df_small, windows=[7])
        
        assert 'Close_ma_7' in df_with_features.columns
        # All values will be NaN since window > data size (7 > 5)
        assert df_with_features['Close_ma_7'].isna().sum() == 5
    
    def test_custom_configuration(self):
        """Test FeatureEngineer with custom configuration."""
        custom_engineer = FeatureEngineer(
            lag_periods=[1, 2, 3],
            rolling_windows=[5, 10],
            rsi_period=10
        )
        
        assert custom_engineer.lag_periods == [1, 2, 3]
        assert custom_engineer.rolling_windows == [5, 10]
        assert custom_engineer.rsi_period == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
