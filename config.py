"""
Configuration module for Gold Price Prediction System.

This module contains all system configuration parameters including:
- Directory paths
- Model training parameters
- Feature engineering settings
- Economic indicator configurations
- Prediction parameters
- Quality monitoring thresholds
"""

import os
from pathlib import Path


class Config:
    """System configuration parameters."""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.resolve()
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Model training parameters
    SEQUENCE_LENGTH = 60  # Number of time steps for LSTM/GRU input
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # Feature engineering parameters
    LAG_PERIODS = [1, 7, 14, 30]
    ROLLING_WINDOWS = [7, 14, 30, 90]
    ROLLING_STD_WINDOWS = [7, 14, 30]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_WINDOW = 20
    BOLLINGER_STD = 2
    
    # Economic indicators (yfinance tickers)
    INDICATORS = {
        'DXY': 'DX-Y.NYB',      # US Dollar Index
        'Oil': 'CL=F',           # Crude Oil Futures
        'Treasury_10Y': '^TNX'   # 10-Year Treasury Yield
    }
    
    # OHLCV data columns
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    TARGET_COLUMN = 'Close'
    
    # Prediction parameters
    DEFAULT_FORECAST_HORIZON = 30  # days
    CONFIDENCE_LEVEL = 0.95
    
    # Quality monitoring thresholds
    MAX_MISSING_PCT = 0.20  # 20% threshold
    DRIFT_THRESHOLD = 0.25   # 25% increase in error triggers alert
    OUTLIER_STD_THRESHOLD = 3.0  # Number of standard deviations
    MAX_FORWARD_FILL_GAP = 3  # Maximum gap days for forward-fill
    MIN_RECORDS_PER_SPLIT = 100  # Minimum records in train/val/test splits
    
    # Normalization method
    NORMALIZATION_METHOD = 'minmax'  # Options: 'minmax', 'zscore'
    
    # Model hyperparameter search spaces
    LSTM_HYPERPARAMS = {
        'units_layer1': [64, 128, 256],
        'units_layer2': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001]
    }
    
    GRU_HYPERPARAMS = {
        'units_layer1': [64, 128, 256],
        'units_layer2': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001]
    }
    
    XGBOOST_HYPERPARAMS = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    RANDOM_FOREST_HYPERPARAMS = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Logging configuration
    LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Model registry
    REGISTRY_FILE = MODEL_DIR / "registry.json"
    
    # Visualization settings
    FIGURE_DPI = 100
    FIGURE_SIZE = (12, 6)
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    
    @classmethod
    def get_model_path(cls, version: str, model_type: str) -> Path:
        """Get the path for a specific model version."""
        model_dir = cls.MODEL_DIR / f"model_{version}"
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    @classmethod
    def get_report_path(cls, report_name: str) -> Path:
        """Get the path for a specific report."""
        return cls.REPORTS_DIR / report_name
    
    @classmethod
    def get_log_path(cls, log_name: str = 'gold_prediction.log') -> Path:
        """Get the path for log files."""
        return cls.LOGS_DIR / log_name
