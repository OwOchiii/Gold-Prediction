"""
Test suite to verify project setup and infrastructure.

This module tests that all directories, configuration, and logging
are properly set up and working correctly.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.logger import get_logger, setup_logging


class TestProjectStructure:
    """Test that required directories exist."""
    
    def test_data_directory_exists(self):
        """Verify data directory exists."""
        assert Config.DATA_DIR.exists()
        assert Config.DATA_DIR.is_dir()
    
    def test_models_directory_exists(self):
        """Verify models directory exists."""
        assert Config.MODEL_DIR.exists()
        assert Config.MODEL_DIR.is_dir()
    
    def test_reports_directory_exists(self):
        """Verify reports directory exists."""
        assert Config.REPORTS_DIR.exists()
        assert Config.REPORTS_DIR.is_dir()
    
    def test_logs_directory_exists(self):
        """Verify logs directory exists."""
        assert Config.LOGS_DIR.exists()
        assert Config.LOGS_DIR.is_dir()


class TestConfiguration:
    """Test configuration parameters."""
    
    def test_config_has_required_attributes(self):
        """Verify Config class has all required attributes."""
        required_attrs = [
            'DATA_DIR', 'MODEL_DIR', 'REPORTS_DIR', 'LOGS_DIR',
            'SEQUENCE_LENGTH', 'TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO',
            'LAG_PERIODS', 'ROLLING_WINDOWS', 'INDICATORS',
            'TARGET_COLUMN', 'REQUIRED_COLUMNS'
        ]
        for attr in required_attrs:
            assert hasattr(Config, attr), f"Config missing attribute: {attr}"
    
    def test_train_val_test_ratios_sum_to_one(self):
        """Verify train/val/test ratios sum to 1.0."""
        total = Config.TRAIN_RATIO + Config.VAL_RATIO + Config.TEST_RATIO
        assert abs(total - 1.0) < 0.001, f"Ratios sum to {total}, expected 1.0"
    
    def test_lag_periods_are_positive(self):
        """Verify lag periods are positive integers."""
        assert all(p > 0 for p in Config.LAG_PERIODS)
    
    def test_rolling_windows_are_positive(self):
        """Verify rolling windows are positive integers."""
        assert all(w > 0 for w in Config.ROLLING_WINDOWS)
    
    def test_required_columns_include_ohlcv(self):
        """Verify required columns include OHLCV data."""
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        actual = set(Config.REQUIRED_COLUMNS)
        assert required.issubset(actual), f"Missing columns: {required - actual}"
    
    def test_target_column_is_valid(self):
        """Verify target column is in required columns."""
        assert Config.TARGET_COLUMN in Config.REQUIRED_COLUMNS


class TestLogging:
    """Test logging configuration."""
    
    def test_logger_creation(self):
        """Verify logger can be created."""
        logger = get_logger('test_logger')
        assert logger is not None
        assert logger.name == 'test_logger'
    
    def test_logger_has_handlers(self):
        """Verify logger has handlers configured."""
        logger = setup_logging('test_handler_logger', log_to_file=False)
        assert len(logger.handlers) > 0
    
    def test_logger_info_level(self):
        """Verify logger accepts INFO level messages."""
        logger = get_logger('test_info_logger')
        try:
            logger.info('Test info message')
            success = True
        except Exception:
            success = False
        assert success, "Logger failed to log INFO message"
    
    def test_log_file_creation(self):
        """Verify log file is created when logging to file."""
        test_log_name = 'test_file_creation'
        logger = setup_logging(test_log_name, log_to_file=True, log_to_console=False)
        logger.info('Test log entry')
        
        log_file = Config.get_log_path(f'{test_log_name}.log')
        assert log_file.exists(), f"Log file not created at {log_file}"
        
        # Clean up test log file
        if log_file.exists():
            log_file.unlink()


class TestConfigMethods:
    """Test Config utility methods."""
    
    def test_get_model_path(self):
        """Verify get_model_path creates correct path structure."""
        version = 'v1.0.0'
        model_type = 'LSTM'
        path = Config.get_model_path(version, model_type)
        
        assert path.exists()
        assert path.is_dir()
        assert f"model_{version}" in str(path)
    
    def test_get_report_path(self):
        """Verify get_report_path returns correct path."""
        report_name = 'test_report.html'
        path = Config.get_report_path(report_name)
        
        assert Config.REPORTS_DIR in path.parents
        assert path.name == report_name
    
    def test_get_log_path(self):
        """Verify get_log_path returns correct path."""
        log_name = 'test.log'
        path = Config.get_log_path(log_name)
        
        assert Config.LOGS_DIR in path.parents
        assert path.name == log_name


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
