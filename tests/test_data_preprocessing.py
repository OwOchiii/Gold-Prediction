"""
Unit tests for data preprocessing and cleaning module.

Tests cover:
- Missing value handling with forward-fill
- Linear interpolation for economic indicators
- Min-max and z-score normalization
- Dataset alignment
- Outlier removal
- Quality report generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_preprocessing import DataPreprocessor, DataQualityReport


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance for testing."""
        return DataPreprocessor(max_forward_fill_gap=3, outlier_std_threshold=3.0)
    
    @pytest.fixture
    def sample_data_with_gaps(self):
        """Create sample data with missing values (small gaps)."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        data = {
            'Close': [100.0, 101.0, np.nan, np.nan, 104.0, 105.0, np.nan, 107.0, 108.0, 109.0],
            'Volume': [1000, 1100, np.nan, 1300, 1400, 1500, np.nan, 1700, 1800, 1900]
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_data_with_large_gaps(self):
        """Create sample data with large missing value gaps."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        data = {
            'Close': [100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0]
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_data_for_normalization(self):
        """Create sample data for normalization testing."""
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        data = {
            'Price': [100.0, 110.0, 120.0, 130.0, 140.0],
            'Volume': [1000, 2000, 3000, 4000, 5000]
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_data_with_outliers(self):
        """Create sample data with outliers."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        # Normal data around 100 with std of 5
        data = {
            'Close': np.random.normal(100, 5, 97).tolist() + [200.0, 10.0, 150.0]  # Add outliers
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    # Test 3.1: Missing value handling with forward-fill
    
    def test_handle_missing_values_small_gaps(self, preprocessor, sample_data_with_gaps):
        """Test forward-fill works correctly for gaps ≤3 days."""
        result = preprocessor.handle_missing_values(sample_data_with_gaps, max_gap=3)
        
        # Check that small gaps are filled
        assert result['Close'].iloc[2] == 101.0, "Gap of 1-2 days should be forward-filled"
        assert result['Close'].iloc[3] == 101.0, "Gap of 1-2 days should be forward-filled"
        assert result['Close'].iloc[6] == 105.0, "Single gap should be forward-filled"
        
        # Verify no NaN in result for Close (all gaps are ≤3)
        assert result['Close'].isna().sum() == 0, "All gaps ≤3 days should be filled"
    
    def test_handle_missing_values_large_gaps(self, preprocessor, sample_data_with_large_gaps):
        """Test that gaps larger than max_gap are not filled."""
        result = preprocessor.handle_missing_values(sample_data_with_large_gaps, max_gap=3)
        
        # Gap is 4 days (indices 1-4), should only fill 3
        assert result['Close'].iloc[1] == 100.0, "First 3 days should be forward-filled"
        assert result['Close'].iloc[2] == 100.0, "First 3 days should be forward-filled"
        assert result['Close'].iloc[3] == 100.0, "First 3 days should be forward-filled"
        assert pd.isna(result['Close'].iloc[4]), "4th day exceeds max_gap, should remain NaN"
    
    def test_handle_missing_values_statistics(self, preprocessor, sample_data_with_gaps):
        """Test that missing value statistics are tracked."""
        preprocessor.reset_statistics()
        result = preprocessor.handle_missing_values(sample_data_with_gaps)
        
        # Check statistics were updated
        assert 'Close' in preprocessor.stats['missing_values_handled']
        assert preprocessor.stats['missing_values_handled']['Close'] > 0
    
    # Test 3.2: Linear interpolation for economic indicators
    
    def test_interpolate_economic_indicators(self, preprocessor):
        """Test linear interpolation fills missing values correctly."""
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        data = {
            'DXY': [100.0, np.nan, np.nan, 104.0, 105.0],
            'Oil': [50.0, 51.0, np.nan, 53.0, 54.0]
        }
        df = pd.DataFrame(data, index=dates)
        
        result = preprocessor.interpolate_economic_indicators(df)
        
        # Check linear interpolation results
        # For DXY: 100 -> 104 should give 101, 102, 103
        assert np.isclose(result['DXY'].iloc[1], 101.0, atol=1.0), "Linear interpolation should produce intermediate values"
        assert np.isclose(result['DXY'].iloc[2], 102.0, atol=1.0), "Linear interpolation should produce intermediate values"
        
        # For Oil: 51 -> 53 should give 52
        assert np.isclose(result['Oil'].iloc[2], 52.0, atol=0.1), "Linear interpolation should produce intermediate values"
        
        # No missing values should remain
        assert result.isna().sum().sum() == 0, "All missing values should be interpolated"
    
    # Test 3.3: Normalization functionality
    
    def test_normalize_features_minmax(self, preprocessor, sample_data_for_normalization):
        """Test min-max normalization produces values in [0, 1]."""
        result, scaling_params = preprocessor.normalize_features(
            sample_data_for_normalization, 
            method='minmax'
        )
        
        # Check that all values are in [0, 1]
        for column in result.columns:
            assert result[column].min() >= 0.0, f"{column} min should be >= 0"
            assert result[column].max() <= 1.0, f"{column} max should be <= 1"
        
        # Check scaling parameters are stored
        assert scaling_params['method'] == 'minmax'
        assert 'Price' in scaling_params['params']
        assert 'min' in scaling_params['params']['Price']
        assert 'max' in scaling_params['params']['Price']
        
        # Verify specific values
        assert np.isclose(result['Price'].iloc[0], 0.0, atol=0.01), "Min value should normalize to 0"
        assert np.isclose(result['Price'].iloc[-1], 1.0, atol=0.01), "Max value should normalize to 1"
    
    def test_normalize_features_zscore(self, preprocessor, sample_data_for_normalization):
        """Test z-score standardization."""
        result, scaling_params = preprocessor.normalize_features(
            sample_data_for_normalization,
            method='zscore'
        )
        
        # Check that mean is approximately 0 and std is approximately 1
        for column in result.columns:
            assert np.isclose(result[column].mean(), 0.0, atol=0.01), f"{column} mean should be ~0"
            assert np.isclose(result[column].std(), 1.0, atol=0.01), f"{column} std should be ~1"
        
        # Check scaling parameters are stored
        assert scaling_params['method'] == 'zscore'
        assert 'mean' in scaling_params['params']['Price']
        assert 'std' in scaling_params['params']['Price']
    
    def test_denormalize_features_minmax(self, preprocessor, sample_data_for_normalization):
        """Test denormalization reverses normalization correctly."""
        original = sample_data_for_normalization.copy()
        
        # Normalize
        normalized, scaling_params = preprocessor.normalize_features(original, method='minmax')
        
        # Denormalize
        denormalized = preprocessor.denormalize_features(normalized, scaling_params)
        
        # Check values are restored
        for column in original.columns:
            assert np.allclose(denormalized[column], original[column], atol=0.01), \
                f"{column} should be restored to original values"
    
    def test_denormalize_features_zscore(self, preprocessor, sample_data_for_normalization):
        """Test denormalization with z-score method."""
        original = sample_data_for_normalization.copy()
        
        # Normalize
        normalized, scaling_params = preprocessor.normalize_features(original, method='zscore')
        
        # Denormalize
        denormalized = preprocessor.denormalize_features(normalized, scaling_params)
        
        # Check values are restored
        for column in original.columns:
            assert np.allclose(denormalized[column], original[column], atol=0.01), \
                f"{column} should be restored to original values"
    
    # Test 3.4: Dataset alignment
    
    def test_align_datasets_inner_join(self, preprocessor):
        """Test dataset alignment with inner join."""
        dates_gold = pd.date_range(start='2020-01-01', periods=10, freq='D')
        dates_dxy = pd.date_range(start='2020-01-03', periods=8, freq='D')
        
        gold_df = pd.DataFrame({'Close': range(10)}, index=dates_gold)
        dxy_df = pd.DataFrame({'DXY': range(100, 108)}, index=dates_dxy)
        
        indicators = {'DXY': dxy_df}
        
        result = preprocessor.align_datasets(gold_df, indicators, strategy='inner')
        
        # Inner join should only keep overlapping dates
        assert len(result) == 8, "Inner join should keep only overlapping dates"
        assert 'Close' in result.columns
        assert 'DXY' in result.columns
        assert result.isna().sum().sum() == 0, "Inner join should have no missing values"
    
    def test_align_datasets_forward_fill(self, preprocessor):
        """Test dataset alignment with forward-fill strategy."""
        dates_gold = pd.date_range(start='2020-01-01', periods=10, freq='D')
        dates_oil = pd.date_range(start='2020-01-01', periods=5, freq='2D')
        
        gold_df = pd.DataFrame({'Close': range(10)}, index=dates_gold)
        oil_df = pd.DataFrame({'Oil': [50, 52, 54, 56, 58]}, index=dates_oil)
        
        indicators = {'Oil': oil_df}
        
        result = preprocessor.align_datasets(gold_df, indicators, strategy='forward_fill')
        
        # Should keep all gold dates
        assert len(result) == 10, "Forward fill should keep all gold dates"
        assert 'Oil' in result.columns
        
        # Check forward-fill worked
        assert result['Oil'].iloc[1] == 50, "Missing values should be forward-filled"
    
    def test_align_datasets_multiple_indicators(self, preprocessor):
        """Test alignment with multiple economic indicators."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        
        gold_df = pd.DataFrame({'Close': range(10)}, index=dates)
        dxy_df = pd.DataFrame({'DXY': range(100, 110)}, index=dates)
        oil_df = pd.DataFrame({'Oil': range(50, 60)}, index=dates)
        treasury_df = pd.DataFrame({'Treasury_10Y': range(2, 12)}, index=dates)
        
        indicators = {
            'DXY': dxy_df,
            'Oil': oil_df,
            'Treasury_10Y': treasury_df
        }
        
        result = preprocessor.align_datasets(gold_df, indicators, strategy='inner')
        
        # Check all columns present
        assert len(result.columns) == 4, "Should have gold + 3 indicators"
        assert 'Close' in result.columns
        assert 'DXY' in result.columns
        assert 'Oil' in result.columns
        assert 'Treasury_10Y' in result.columns
    
    # Test 3.5: Outlier removal
    
    def test_remove_outliers_with_known_data(self, preprocessor, sample_data_with_outliers):
        """Test outlier removal using z-score method."""
        original_count = len(sample_data_with_outliers)
        
        result = preprocessor.remove_outliers(sample_data_with_outliers, n_std=3.0)
        
        # Should have fewer records
        assert len(result) < original_count, "Outliers should be removed"
        
        # Extreme values (200, 10, 150) should be removed
        # Normal data is around 100 ± 15 (3 std)
        assert result['Close'].max() < 180, "Extreme high values should be removed"
        assert result['Close'].min() > 20, "Extreme low values should be removed"
    
    def test_remove_outliers_statistics(self, preprocessor, sample_data_with_outliers):
        """Test that outlier statistics are tracked."""
        preprocessor.reset_statistics()
        result = preprocessor.remove_outliers(sample_data_with_outliers)
        
        # Check statistics
        assert 'Close' in preprocessor.stats['outliers_removed']
        assert preprocessor.stats['outliers_removed']['Close'] > 0
    
    def test_remove_outliers_no_outliers(self, preprocessor):
        """Test outlier removal when no outliers exist."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        data = {'Close': [100.0 + i for i in range(10)]}  # No outliers
        df = pd.DataFrame(data, index=dates)
        
        result = preprocessor.remove_outliers(df, n_std=3.0)
        
        # Should keep all records
        assert len(result) == len(df), "Should keep all records when no outliers"
    
    # Test 3.6: Data quality reporting
    
    def test_generate_quality_report_structure(self, preprocessor):
        """Test quality report generation returns correct structure."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({'Close': range(10), 'Volume': range(100, 110)}, index=dates)
        
        report = preprocessor.generate_quality_report(df, original_record_count=10)
        
        # Check report structure
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 10
        assert isinstance(report.missing_values_handled, dict)
        assert isinstance(report.outliers_removed, dict)
        assert isinstance(report.date_range, tuple)
        assert isinstance(report.data_quality_score, float)
        assert 0 <= report.data_quality_score <= 100
    
    def test_generate_quality_report_score_calculation(self, preprocessor):
        """Test data quality score calculation."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': range(100),
            'Open': range(100, 200),
            'High': range(200, 300),
            'Low': range(50, 150),
            'Volume': range(1000, 1100)
        }, index=dates)
        
        # Clean data with no preprocessing
        preprocessor.reset_statistics()
        report = preprocessor.generate_quality_report(df, original_record_count=100)
        
        # High quality score expected (no missing, no outliers removed)
        assert report.data_quality_score > 80, "Clean data should have high quality score"
    
    def test_generate_quality_report_with_processing(self, preprocessor, sample_data_with_gaps):
        """Test quality report after preprocessing operations."""
        preprocessor.reset_statistics()
        
        # Process data
        filled = preprocessor.handle_missing_values(sample_data_with_gaps)
        
        # Generate report
        report = preprocessor.generate_quality_report(filled, original_record_count=len(sample_data_with_gaps))
        
        # Check that missing values were tracked
        assert sum(report.missing_values_handled.values()) > 0
        assert report.total_records == len(filled)
    
    # Edge cases and error handling
    
    def test_normalize_invalid_method(self, preprocessor, sample_data_for_normalization):
        """Test that invalid normalization method raises error."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            preprocessor.normalize_features(sample_data_for_normalization, method='invalid')
    
    def test_align_datasets_invalid_strategy(self, preprocessor):
        """Test that invalid alignment strategy raises error."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        gold_df = pd.DataFrame({'Close': range(10)}, index=dates)
        dxy_df = pd.DataFrame({'DXY': range(100, 110)}, index=dates)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            preprocessor.align_datasets(gold_df, {'DXY': dxy_df}, strategy='invalid')
    
    def test_handle_missing_values_preserves_original(self, preprocessor, sample_data_with_gaps):
        """Test that original dataframe is not modified."""
        original = sample_data_with_gaps.copy()
        original_missing = original.isna().sum().sum()
        
        result = preprocessor.handle_missing_values(sample_data_with_gaps)
        
        # Original should be unchanged
        assert sample_data_with_gaps.isna().sum().sum() == original_missing
    
    def test_reset_statistics(self, preprocessor):
        """Test that reset_statistics clears all tracked statistics."""
        # Add some statistics
        preprocessor.stats['missing_values_handled']['test'] = 10
        preprocessor.stats['outliers_removed']['test'] = 5
        
        # Reset
        preprocessor.reset_statistics()
        
        # Check cleared
        assert len(preprocessor.stats['missing_values_handled']) == 0
        assert len(preprocessor.stats['outliers_removed']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
