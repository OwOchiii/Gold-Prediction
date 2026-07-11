"""
Feature Engineering Module

This module provides functionality to create derived features from raw data
to improve model performance.

Classes:
    - FeatureEngineer: Main class for feature engineering operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available. Technical indicators will use manual calculations.")

from src.logger import get_logger
from config import Config

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering operations to create derived features.
    
    This class provides methods for creating lag features, rolling statistics,
    technical indicators, interaction features, and temporal features.
    """
    
    def __init__(self, 
                 lag_periods: Optional[List[int]] = None,
                 rolling_windows: Optional[List[int]] = None,
                 rolling_std_windows: Optional[List[int]] = None,
                 rsi_period: Optional[int] = None,
                 macd_fast: Optional[int] = None,
                 macd_slow: Optional[int] = None,
                 macd_signal: Optional[int] = None,
                 bollinger_window: Optional[int] = None,
                 bollinger_std: Optional[int] = None):
        """
        Initialize FeatureEngineer with configuration parameters.
        
        Args:
            lag_periods: List of lag periods for lag features (default: from Config)
            rolling_windows: List of window sizes for rolling mean (default: from Config)
            rolling_std_windows: List of window sizes for rolling std (default: from Config)
            rsi_period: Period for RSI calculation (default: from Config)
            macd_fast: Fast period for MACD (default: from Config)
            macd_slow: Slow period for MACD (default: from Config)
            macd_signal: Signal period for MACD (default: from Config)
            bollinger_window: Window for Bollinger Bands (default: from Config)
            bollinger_std: Standard deviation multiplier for Bollinger Bands (default: from Config)
        """
        self.lag_periods = lag_periods or Config.LAG_PERIODS
        self.rolling_windows = rolling_windows or Config.ROLLING_WINDOWS
        self.rolling_std_windows = rolling_std_windows or Config.ROLLING_STD_WINDOWS
        self.rsi_period = rsi_period or Config.RSI_PERIOD
        self.macd_fast = macd_fast or Config.MACD_FAST
        self.macd_slow = macd_slow or Config.MACD_SLOW
        self.macd_signal = macd_signal or Config.MACD_SIGNAL
        self.bollinger_window = bollinger_window or Config.BOLLINGER_WINDOW
        self.bollinger_std = bollinger_std or Config.BOLLINGER_STD
        
        logger.info("FeatureEngineer initialized")
        logger.info(f"Lag periods: {self.lag_periods}")
        logger.info(f"Rolling windows (mean): {self.rolling_windows}")
        logger.info(f"Rolling windows (std): {self.rolling_std_windows}")
        logger.info(f"RSI period: {self.rsi_period}")
        logger.info(f"MACD parameters: fast={self.macd_fast}, slow={self.macd_slow}, signal={self.macd_signal}")
        logger.info(f"Bollinger Bands: window={self.bollinger_window}, std={self.bollinger_std}")
    
    def create_lag_features(self, df: pd.DataFrame, 
                           column: str = 'Close',
                           lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lag features for specified column.
        
        Generates lag features by shifting the target column by specified periods.
        Useful for capturing historical price patterns.
        
        Args:
            df: DataFrame containing the column to create lags from
            column: Column name to create lag features from (default: 'Close')
            lags: List of lag periods (default: uses instance setting)
        
        Returns:
            DataFrame with added lag feature columns
            
        Requirements: 3.1
        """
        if lags is None:
            lags = self.lag_periods
        
        logger.info(f"Creating lag features for {column} with periods {lags}")
        
        df_copy = df.copy()
        
        for lag in lags:
            feature_name = f"{column}_lag_{lag}"
            df_copy[feature_name] = df_copy[column].shift(lag)
            logger.debug(f"Created {feature_name}")
        
        # Log how many NaN values were introduced
        nan_count = df_copy[[f"{column}_lag_{lag}" for lag in lags]].isna().sum().sum()
        logger.info(f"Created {len(lags)} lag features ({nan_count} NaN values introduced)")
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame,
                               column: str = 'Close',
                               windows: Optional[List[int]] = None,
                               std_windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create rolling mean and standard deviation features.
        
        Calculates rolling statistics to capture trend and volatility patterns.
        
        Args:
            df: DataFrame containing the column to calculate rolling stats from
            column: Column name to calculate rolling features from (default: 'Close')
            windows: List of window sizes for rolling mean (default: uses instance setting)
            std_windows: List of window sizes for rolling std (default: uses instance setting)
        
        Returns:
            DataFrame with added rolling feature columns
            
        Requirements: 3.2, 3.3
        """
        if windows is None:
            windows = self.rolling_windows
        if std_windows is None:
            std_windows = self.rolling_std_windows
        
        logger.info(f"Creating rolling features for {column}")
        logger.info(f"Rolling mean windows: {windows}")
        logger.info(f"Rolling std windows: {std_windows}")
        
        df_copy = df.copy()
        
        # Create rolling mean features
        for window in windows:
            feature_name = f"{column}_ma_{window}"
            df_copy[feature_name] = df_copy[column].rolling(window=window).mean()
            logger.debug(f"Created {feature_name}")
        
        # Create rolling standard deviation features
        for window in std_windows:
            feature_name = f"{column}_std_{window}"
            df_copy[feature_name] = df_copy[column].rolling(window=window).std()
            logger.debug(f"Created {feature_name}")
        
        total_features = len(windows) + len(std_windows)
        nan_count = df_copy[[f"{column}_ma_{w}" for w in windows] + 
                            [f"{column}_std_{w}" for w in std_windows]].isna().sum().sum()
        logger.info(f"Created {total_features} rolling features ({nan_count} NaN values introduced)")
        
        return df_copy
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators: RSI, MACD, and Bollinger Bands.
        
        Uses pandas-ta library if available, otherwise falls back to manual calculations.
        
        Args:
            df: DataFrame containing OHLCV data
        
        Returns:
            DataFrame with added technical indicator columns
            
        Requirements: 3.4
        """
        logger.info("Creating technical indicators (RSI, MACD, Bollinger Bands)")
        
        df_copy = df.copy()
        
        # Ensure we have Close column
        if 'Close' not in df_copy.columns:
            logger.error("Close column not found in DataFrame")
            raise ValueError("Close column required for technical indicators")
        
        # Calculate RSI
        df_copy = self._calculate_rsi(df_copy)
        
        # Calculate MACD
        df_copy = self._calculate_macd(df_copy)
        
        # Calculate Bollinger Bands
        df_copy = self._calculate_bollinger_bands(df_copy)
        
        logger.info("Technical indicators created successfully")
        
        return df_copy
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures momentum by comparing recent gains to recent losses.
        Values range from 0 to 100, with >70 indicating overbought and <30 oversold.
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            DataFrame with RSI column added
        """
        logger.debug(f"Calculating RSI with period {self.rsi_period}")
        
        if PANDAS_TA_AVAILABLE:
            # Use pandas-ta
            df[f'RSI_{self.rsi_period}'] = ta.rsi(df['Close'], length=self.rsi_period)
        else:
            # Manual calculation
            close_delta = df['Close'].diff()
            
            # Separate gains and losses
            gain = close_delta.where(close_delta > 0, 0)
            loss = -close_delta.where(close_delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df[f'RSI_{self.rsi_period}'] = 100 - (100 / (1 + rs))
        
        logger.debug(f"RSI_{self.rsi_period} created")
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        MACD shows relationship between two moving averages of prices.
        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            DataFrame with MACD columns added
        """
        logger.debug(f"Calculating MACD with parameters: fast={self.macd_fast}, slow={self.macd_slow}, signal={self.macd_signal}")
        
        if PANDAS_TA_AVAILABLE:
            # Use pandas-ta
            macd_result = ta.macd(df['Close'], 
                                 fast=self.macd_fast, 
                                 slow=self.macd_slow, 
                                 signal=self.macd_signal)
            if macd_result is not None:
                df['MACD'] = macd_result[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
                df['MACD_signal'] = macd_result[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
                df['MACD_diff'] = macd_result[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        else:
            # Manual calculation
            ema_fast = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
            
            df['MACD'] = ema_fast - ema_slow
            df['MACD_signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
            df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        logger.debug("MACD, MACD_signal, MACD_diff created")
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of a middle band (SMA) and upper/lower bands
        that are standard deviations away from the middle band.
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        logger.debug(f"Calculating Bollinger Bands with window={self.bollinger_window}, std={self.bollinger_std}")
        
        if PANDAS_TA_AVAILABLE:
            # Use pandas-ta
            bbands = ta.bbands(df['Close'], 
                              length=self.bollinger_window, 
                              std=self.bollinger_std)
            if bbands is not None:
                df['BB_lower'] = bbands[f'BBL_{self.bollinger_window}_{self.bollinger_std}.0']
                df['BB_middle'] = bbands[f'BBM_{self.bollinger_window}_{self.bollinger_std}.0']
                df['BB_upper'] = bbands[f'BBU_{self.bollinger_window}_{self.bollinger_std}.0']
        else:
            # Manual calculation
            df['BB_middle'] = df['Close'].rolling(window=self.bollinger_window).mean()
            rolling_std = df['Close'].rolling(window=self.bollinger_window).std()
            df['BB_upper'] = df['BB_middle'] + (rolling_std * self.bollinger_std)
            df['BB_lower'] = df['BB_middle'] - (rolling_std * self.bollinger_std)
        
        logger.debug("BB_lower, BB_middle, BB_upper created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between gold prices and economic indicators.
        
        Generates features that capture relationships between gold and other economic variables:
        - Gold/Oil ratio
        - Gold/DXY correlation (rolling)
        - Gold vs Treasury yield features
        
        Args:
            df: DataFrame containing gold prices and economic indicators
        
        Returns:
            DataFrame with added interaction feature columns
            
        Requirements: 3.5
        """
        logger.info("Creating interaction features")
        
        df_copy = df.copy()
        features_created = []
        
        # Gold/Oil ratio
        if 'Close' in df_copy.columns and 'Oil' in df_copy.columns:
            df_copy['Gold_Oil_ratio'] = df_copy['Close'] / df_copy['Oil']
            features_created.append('Gold_Oil_ratio')
            logger.debug("Created Gold_Oil_ratio")
        else:
            logger.warning("Cannot create Gold_Oil_ratio: Close or Oil column missing")
        
        # Gold/DXY correlation (30-day rolling)
        if 'Close' in df_copy.columns and 'DXY' in df_copy.columns:
            df_copy['Gold_DXY_corr'] = df_copy['Close'].rolling(window=30).corr(df_copy['DXY'])
            features_created.append('Gold_DXY_corr')
            logger.debug("Created Gold_DXY_corr (30-day rolling correlation)")
        else:
            logger.warning("Cannot create Gold_DXY_corr: Close or DXY column missing")
        
        # Gold vs Treasury yield spread
        if 'Close' in df_copy.columns and 'Treasury_10Y' in df_copy.columns:
            # Normalize gold price change rate vs treasury yield change
            df_copy['Gold_Treasury_spread'] = df_copy['Close'].pct_change() - df_copy['Treasury_10Y'].pct_change()
            features_created.append('Gold_Treasury_spread')
            logger.debug("Created Gold_Treasury_spread")
        else:
            logger.warning("Cannot create Gold_Treasury_spread: Close or Treasury_10Y column missing")
        
        logger.info(f"Created {len(features_created)} interaction features: {features_created}")
        
        return df_copy
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from date index.
        
        Creates features based on calendar information that may affect gold prices:
        - day_of_week (0=Monday, 6=Sunday)
        - month (1-12)
        - quarter (1-4)
        - year
        - is_quarter_end (boolean)
        - is_year_end (boolean)
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with added temporal feature columns
            
        Requirements: 3.6
        """
        logger.info("Creating temporal features")
        
        df_copy = df.copy()
        
        # Verify we have a DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            logger.error("DataFrame must have DatetimeIndex for temporal features")
            raise ValueError("DataFrame must have DatetimeIndex for temporal features")
        
        # Extract temporal features
        df_copy['day_of_week'] = df_copy.index.dayofweek
        df_copy['month'] = df_copy.index.month
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['year'] = df_copy.index.year
        df_copy['is_quarter_end'] = df_copy.index.is_quarter_end.astype(int)
        df_copy['is_year_end'] = df_copy.index.is_year_end.astype(int)
        
        logger.info("Created 6 temporal features: day_of_week, month, quarter, year, is_quarter_end, is_year_end")
        
        return df_copy
    
    def build_feature_set(self, df: pd.DataFrame,
                         create_lags: bool = True,
                         create_rolling: bool = True,
                         create_technical: bool = True,
                         create_interactions: bool = True,
                         create_temporal: bool = True,
                         handle_nan: bool = True) -> pd.DataFrame:
        """
        Build complete feature set by orchestrating all feature engineering steps.
        
        Applies all feature engineering methods in sequence and optionally handles
        NaN values introduced by feature creation.
        
        Args:
            df: DataFrame containing raw OHLCV data and economic indicators
            create_lags: Whether to create lag features (default: True)
            create_rolling: Whether to create rolling features (default: True)
            create_technical: Whether to create technical indicators (default: True)
            create_interactions: Whether to create interaction features (default: True)
            create_temporal: Whether to create temporal features (default: True)
            handle_nan: Whether to drop rows with NaN values after feature creation (default: True)
        
        Returns:
            DataFrame with complete feature set
            
        Requirements: 3.7
        """
        logger.info("=" * 60)
        logger.info("BUILDING COMPLETE FEATURE SET")
        logger.info("=" * 60)
        logger.info(f"Input shape: {df.shape}")
        
        df_features = df.copy()
        initial_columns = len(df_features.columns)
        
        # Track features created
        features_summary = []
        
        # Create lag features
        if create_lags:
            df_features = self.create_lag_features(df_features)
            features_summary.append(f"Lag features: {len(self.lag_periods)}")
        
        # Create rolling features
        if create_rolling:
            df_features = self.create_rolling_features(df_features)
            features_summary.append(f"Rolling features: {len(self.rolling_windows) + len(self.rolling_std_windows)}")
        
        # Create technical indicators
        if create_technical:
            df_features = self.create_technical_indicators(df_features)
            features_summary.append("Technical indicators: RSI, MACD (3 features), Bollinger Bands (3 features)")
        
        # Create interaction features
        if create_interactions:
            df_features = self.create_interaction_features(df_features)
            features_summary.append("Interaction features: Gold/Oil ratio, Gold/DXY corr, Gold/Treasury spread")
        
        # Create temporal features
        if create_temporal:
            df_features = self.create_temporal_features(df_features)
            features_summary.append("Temporal features: 6 (day_of_week, month, quarter, year, is_quarter_end, is_year_end)")
        
        # Handle NaN values
        if handle_nan:
            logger.info("Handling NaN values introduced by feature engineering")
            rows_before = len(df_features)
            nan_count_before = df_features.isna().sum().sum()
            logger.info(f"NaN values before handling: {nan_count_before}")
            
            # Drop rows with any NaN values
            df_features = df_features.dropna()
            
            rows_after = len(df_features)
            rows_dropped = rows_before - rows_after
            logger.info(f"Dropped {rows_dropped} rows containing NaN values ({rows_dropped/rows_before*100:.2f}%)")
        
        # Final summary
        final_columns = len(df_features.columns)
        features_added = final_columns - initial_columns
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)
        for summary in features_summary:
            logger.info(f"  - {summary}")
        logger.info(f"Total features added: {features_added}")
        logger.info(f"Final shape: {df_features.shape}")
        logger.info(f"Final columns: {list(df_features.columns)}")
        logger.info("=" * 60)
        
        return df_features


# Utility function for quick feature engineering
def engineer_features(df: pd.DataFrame,
                     config: Optional[Config] = None,
                     **kwargs) -> pd.DataFrame:
    """
    Convenience function to quickly engineer features using default configuration.
    
    Args:
        df: DataFrame containing raw OHLCV data and economic indicators
        config: Configuration object (uses default Config if not provided)
        **kwargs: Additional arguments passed to build_feature_set()
    
    Returns:
        DataFrame with complete feature set
    
    Example:
        >>> df_with_features = engineer_features(raw_df)
    """
    engineer = FeatureEngineer()
    return engineer.build_feature_set(df, **kwargs)
