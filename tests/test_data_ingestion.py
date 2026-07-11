"""
Unit tests for data ingestion module.

Tests cover:
- CSV loading with valid and invalid files
- OHLCV constraint validation
- Chronological order validation
- Duplicate detection
- Missing value detection
- Economic indicators fetching
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.data_ingestion import DataIngestionManager, ValidationResult


class TestDataIngestionManager:
    """Test suite for DataIngestionManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create DataIngestionManager instance for testing."""
        return DataIngestionManager()
    
    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create valid OHLCV DataFrame for testing."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        return df
    
    # Test CSV loading (Subtask 2.1)
    def test_load_csv_valid_file(self, manager):
        """Test loading valid CSV file."""
        # Create temporary CSV file
        csv_content = """Date;Open;High;Low;Close;Volume
2020.01.01 00:00;100.0;105.0;99.0;103.0;1000
2020.01.02 00:00;101.0;106.0;100.0;104.0;1100
2020.01.03 00:00;102.0;107.0;101.0;105.0;1200"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = manager.load_csv(temp_path)
            
            # Verify DataFrame properties
            assert len(df) == 3
            assert isinstance(df.index, pd.DatetimeIndex)
            assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert df['Close'].iloc[0] == 103.0
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_file_not_found(self, manager):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.load_csv('nonexistent_file.csv')
    
    def test_load_csv_empty_file(self, manager):
        """Test loading empty CSV file."""
        csv_content = """Date;Open;High;Low;Close;Volume"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = manager.load_csv(temp_path)
            
            # Should return empty DataFrame with correct structure
            assert len(df) == 0
            assert isinstance(df.index, pd.DatetimeIndex)
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_invalid_date_format(self, manager):
        """Test loading CSV with invalid date format."""
        csv_content = """Date;Open;High;Low;Close;Volume
invalid_date;100.0;105.0;99.0;103.0;1000
2020.01.02 00:00;101.0;106.0;100.0;104.0;1100"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            # Should load but log warnings about failed date parsing
            df = manager.load_csv(temp_path)
            
            # Should have parsed the valid date
            assert len(df) >= 1
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_custom_separator(self, manager):
        """Test loading CSV with comma separator."""
        csv_content = """Date,Open,High,Low,Close,Volume
2020.01.01 00:00,100.0,105.0,99.0,103.0,1000
2020.01.02 00:00,101.0,106.0,100.0,104.0,1100"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = manager.load_csv(temp_path, separator=',')
            
            assert len(df) == 2
            assert df['Close'].iloc[0] == 103.0
        finally:
            os.unlink(temp_path)
    
    # Test OHLCV validation (Subtask 2.2)
    def test_validate_ohlcv_valid_data(self, manager, valid_ohlcv_data):
        """Test validation passes for valid OHLCV data."""
        result = manager.validate_ohlcv_data(valid_ohlcv_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_ohlcv_missing_columns(self, manager):
        """Test validation fails when required columns are missing."""
        df = pd.DataFrame({'Open': [100], 'High': [105]})
        result = manager.validate_ohlcv_data(df)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert 'Missing required columns' in result.errors[0]
    
    def test_validate_ohlcv_high_less_than_low(self, manager, valid_ohlcv_data):
        """Test validation fails when High < Low (Requirement 1.7)."""
        # Create violation: High < Low
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'High'] = 90.0  # High < Low
        
        result = manager.validate_ohlcv_data(invalid_data)
        
        assert not result.is_valid
        assert any('High < Low' in error for error in result.errors)
    
    def test_validate_ohlcv_close_outside_range(self, manager, valid_ohlcv_data):
        """Test validation fails when Close is outside [Low, High] (Requirement 1.8)."""
        # Create violation: Close > High
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = 120.0  # Close > High
        
        result = manager.validate_ohlcv_data(invalid_data)
        
        assert not result.is_valid
        assert any('Close outside' in error for error in result.errors)
    
    def test_validate_ohlcv_open_outside_range(self, manager, valid_ohlcv_data):
        """Test validation fails when Open is outside [Low, High] (Requirement 1.8)."""
        # Create violation: Open < Low
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Open'] = 90.0  # Open < Low
        
        result = manager.validate_ohlcv_data(invalid_data)
        
        assert not result.is_valid
        assert any('Open outside' in error for error in result.errors)
    
    def test_validate_ohlcv_multiple_violations(self, manager, valid_ohlcv_data):
        """Test validation detects multiple violations in the same dataset."""
        # Create multiple violations
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'High'] = 90.0  # High < Low
        invalid_data.loc[invalid_data.index[1], 'Close'] = 120.0  # Close > High
        invalid_data.loc[invalid_data.index[2], 'Open'] = 95.0  # Open < Low
        
        result = manager.validate_ohlcv_data(invalid_data)
        
        assert not result.is_valid
        # Should have at least 3 errors
        assert len(result.errors) >= 3
    
    def test_validate_ohlcv_with_missing_values(self, manager, valid_ohlcv_data):
        """Test validation warns about missing values but doesn't fail."""
        data_with_missing = valid_ohlcv_data.copy()
        data_with_missing.loc[data_with_missing.index[0], 'Close'] = np.nan
        
        result = manager.validate_ohlcv_data(data_with_missing)
        
        # Should have warnings but may still be valid if no constraint violations
        assert len(result.warnings) > 0
        assert any('Missing values' in warning for warning in result.warnings)
    
    def test_validate_ohlcv_negative_values(self, manager, valid_ohlcv_data):
        """Test validation warns about negative price values."""
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = -10.0
        
        result = manager.validate_ohlcv_data(invalid_data)
        
        # Should have warning about negative values
        assert len(result.warnings) > 0
        assert any('Negative values' in warning for warning in result.warnings)
    
    # Test chronological order validation (Subtask 2.3)
    def test_validate_chronological_order_valid(self, manager, valid_ohlcv_data):
        """Test chronological order validation passes for sorted dates."""
        result = manager.validate_chronological_order(valid_ohlcv_data)
        
        assert result is True
    
    def test_validate_chronological_order_invalid(self, manager, valid_ohlcv_data):
        """Test chronological order validation fails for unsorted dates."""
        # Reverse the index to make it non-chronological
        invalid_data = valid_ohlcv_data.iloc[::-1]
        
        result = manager.validate_chronological_order(invalid_data)
        
        assert result is False
    
    def test_validate_chronological_order_non_datetime_index(self, manager):
        """Test chronological order validation handles non-DatetimeIndex."""
        df = pd.DataFrame({'Open': [100, 101]}, index=[0, 1])
        
        result = manager.validate_chronological_order(df)
        
        assert result is False
    
    # Test duplicate detection (Subtask 2.3)
    def test_check_duplicates_none(self, manager, valid_ohlcv_data):
        """Test duplicate detection returns empty list when no duplicates."""
        duplicates = manager.check_duplicates(valid_ohlcv_data)
        
        assert len(duplicates) == 0
    
    def test_check_duplicates_found(self, manager):
        """Test duplicate detection identifies duplicate dates."""
        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1)]
        data = {
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
        
        duplicates = manager.check_duplicates(df)
        
        assert len(duplicates) == 1
        assert duplicates[0] == datetime(2020, 1, 1)
    
    def test_check_duplicates_multiple(self, manager):
        """Test duplicate detection identifies multiple duplicate dates."""
        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1), 
                 datetime(2020, 1, 3), datetime(2020, 1, 2)]
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'Close': [103.0, 104.0, 105.0, 106.0, 107.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
        
        duplicates = manager.check_duplicates(df)
        
        assert len(duplicates) == 2
        assert datetime(2020, 1, 1) in duplicates
        assert datetime(2020, 1, 2) in duplicates
    
    # Test missing value detection (Subtask 2.5)
    def test_detect_missing_values_none(self, manager, valid_ohlcv_data):
        """Test missing value detection returns empty dict when no missing values."""
        missing = manager.detect_missing_values(valid_ohlcv_data)
        
        assert len(missing) == 0
    
    def test_detect_missing_values_found(self, manager, valid_ohlcv_data):
        """Test missing value detection identifies missing values and affected records."""
        data_with_missing = valid_ohlcv_data.copy()
        data_with_missing.loc[data_with_missing.index[0], 'Close'] = np.nan
        data_with_missing.loc[data_with_missing.index[2], 'Close'] = np.nan
        data_with_missing.loc[data_with_missing.index[1], 'Volume'] = np.nan
        
        missing = manager.detect_missing_values(data_with_missing)
        
        assert 'Close' in missing
        assert len(missing['Close']) == 2
        assert 'Volume' in missing
        assert len(missing['Volume']) == 1
    
    def test_detect_missing_values_all_columns(self, manager, valid_ohlcv_data):
        """Test missing value detection across all OHLCV columns."""
        data_with_missing = valid_ohlcv_data.copy()
        # Add missing values to each column
        data_with_missing.loc[data_with_missing.index[0], 'Open'] = np.nan
        data_with_missing.loc[data_with_missing.index[1], 'High'] = np.nan
        data_with_missing.loc[data_with_missing.index[2], 'Low'] = np.nan
        data_with_missing.loc[data_with_missing.index[3], 'Close'] = np.nan
        data_with_missing.loc[data_with_missing.index[4], 'Volume'] = np.nan
        
        missing = manager.detect_missing_values(data_with_missing)
        
        # All OHLCV columns should have missing values
        assert 'Open' in missing
        assert 'High' in missing
        assert 'Low' in missing
        assert 'Close' in missing
        assert 'Volume' in missing
        assert len(missing) == 5
    
    # Test economic indicators (Subtask 2.4)
    def test_load_economic_indicators_structure(self, manager, mocker):
        """Test economic indicators loading returns correct structure (mocked)."""
        # Mock yfinance download to avoid actual API calls
        mock_data = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        
        mocker.patch('yfinance.download', return_value=mock_data)
        
        tickers = {'DXY': 'DX-Y.NYB', 'Oil': 'CL=F'}
        
        result = manager.load_economic_indicators(
            tickers=tickers,
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Verify structure
        assert 'DXY' in result
        assert 'Oil' in result
        assert isinstance(result['DXY'], pd.DataFrame)
        assert isinstance(result['Oil'], pd.DataFrame)
        assert 'DXY' in result['DXY'].columns
        assert 'Oil' in result['Oil'].columns
        assert isinstance(result['DXY'].index, pd.DatetimeIndex)
    
    def test_load_economic_indicators_empty_data(self, manager, mocker):
        """Test economic indicators loading handles empty data."""
        # Mock yfinance download to return empty DataFrame
        mock_data = pd.DataFrame()
        
        mocker.patch('yfinance.download', return_value=mock_data)
        
        tickers = {'DXY': 'DX-Y.NYB'}
        
        with pytest.raises(ValueError, match='Failed to download data'):
            manager.load_economic_indicators(
                tickers=tickers,
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
    
    def test_load_economic_indicators_api_failure(self, manager, mocker):
        """Test economic indicators loading handles API failures."""
        # Mock yfinance download to raise exception
        mocker.patch('yfinance.download', side_effect=Exception('API Error'))
        
        tickers = {'DXY': 'DX-Y.NYB'}
        
        with pytest.raises(ValueError, match='Failed to load economic indicator'):
            manager.load_economic_indicators(
                tickers=tickers,
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
    
    # Test validation summary
    def test_get_validation_summary(self, manager, valid_ohlcv_data):
        """Test comprehensive validation summary generation."""
        summary = manager.get_validation_summary(valid_ohlcv_data)
        
        assert 'total_records' in summary
        assert summary['total_records'] == 10
        assert 'date_range' in summary
        assert 'ohlcv_validation' in summary
        assert 'chronological_order' in summary
        assert 'duplicates' in summary
        assert 'missing_values' in summary
        assert summary['chronological_order'] is True
        assert len(summary['duplicates']) == 0


class TestValidationResult:
    """Test ValidationResult data class."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult instance."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Test warning']
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.warnings[0] == 'Test warning'
