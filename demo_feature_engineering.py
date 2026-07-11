"""
Demonstration script for Feature Engineering Module

This script demonstrates the feature engineering capabilities by:
1. Loading sample gold price data
2. Loading economic indicators
3. Creating all types of features
4. Showing the complete feature set
"""

import pandas as pd
import numpy as np
from datetime import datetime

from src.feature_engineering import FeatureEngineer, engineer_features
from src.logger import get_logger
from config import Config

logger = get_logger(__name__)


def main():
    """Run feature engineering demonstration."""
    
    print("=" * 80)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create sample data for demonstration
    print("Step 1: Creating sample gold price data with economic indicators...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate synthetic data with realistic patterns
    base_price = 1800
    trend = np.linspace(0, 100, len(dates))
    seasonal = 50 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
    noise = np.random.randn(len(dates)) * 15
    close_prices = base_price + trend + seasonal + noise
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(len(dates)) * 8,
        'High': close_prices + np.abs(np.random.randn(len(dates)) * 12),
        'Low': close_prices - np.abs(np.random.randn(len(dates)) * 12),
        'Close': close_prices,
        'Volume': np.random.randint(50000, 150000, len(dates)),
        'DXY': 102 + np.random.randn(len(dates)) * 3,
        'Oil': 75 + np.random.randn(len(dates)) * 8,
        'Treasury_10Y': 3.5 + np.random.randn(len(dates)) * 0.3
    }, index=dates)
    
    print(f"Created sample data with {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Initial columns: {list(df.columns)}")
    print()
    
    # Initialize Feature Engineer
    print("Step 2: Initializing FeatureEngineer...")
    engineer = FeatureEngineer()
    print()
    
    # Demonstrate each feature type
    print("Step 3: Creating Lag Features...")
    df_with_lags = engineer.create_lag_features(df.copy())
    lag_features = [col for col in df_with_lags.columns if 'lag' in col]
    print(f"Lag features created: {lag_features}")
    print(f"Example values (row 30):")
    for feature in lag_features:
        print(f"  {feature}: {df_with_lags[feature].iloc[30]:.2f}")
    print()
    
    print("Step 4: Creating Rolling Statistics Features...")
    df_with_rolling = engineer.create_rolling_features(df.copy())
    rolling_features = [col for col in df_with_rolling.columns if 'ma_' in col or 'std_' in col]
    print(f"Rolling features created: {len(rolling_features)}")
    print(f"Features: {rolling_features}")
    print(f"Example - Close_ma_7 at row 100: {df_with_rolling['Close_ma_7'].iloc[100]:.2f}")
    print(f"Example - Close_std_7 at row 100: {df_with_rolling['Close_std_7'].iloc[100]:.2f}")
    print()
    
    print("Step 5: Creating Technical Indicators...")
    df_with_tech = engineer.create_technical_indicators(df.copy())
    tech_features = ['RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_middle', 'BB_lower']
    print(f"Technical indicators created:")
    for feature in tech_features:
        if feature in df_with_tech.columns:
            value = df_with_tech[feature].iloc[100]
            if pd.notna(value):
                print(f"  {feature}: {value:.2f}")
    print()
    
    print("Step 6: Creating Interaction Features...")
    df_with_interactions = engineer.create_interaction_features(df.copy())
    interaction_features = ['Gold_Oil_ratio', 'Gold_DXY_corr', 'Gold_Treasury_spread']
    print(f"Interaction features created:")
    for feature in interaction_features:
        if feature in df_with_interactions.columns:
            value = df_with_interactions[feature].iloc[100]
            if pd.notna(value):
                print(f"  {feature}: {value:.4f}")
    print()
    
    print("Step 7: Creating Temporal Features...")
    df_with_temporal = engineer.create_temporal_features(df.copy())
    temporal_features = ['day_of_week', 'month', 'quarter', 'year', 'is_quarter_end', 'is_year_end']
    print(f"Temporal features created:")
    print(f"Example (row 100, date {df_with_temporal.index[100]}):")
    for feature in temporal_features:
        print(f"  {feature}: {df_with_temporal[feature].iloc[100]}")
    print()
    
    # Build complete feature set
    print("Step 8: Building Complete Feature Set...")
    df_complete = engineer.build_feature_set(df, handle_nan=True)
    print()
    
    print("=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {df_complete.shape}")
    print(f"Features added: {df_complete.shape[1] - df.shape[1]}")
    print(f"Records retained: {df_complete.shape[0]} ({df_complete.shape[0]/df.shape[0]*100:.1f}%)")
    print()
    
    # Show sample of final data
    print("Sample of final feature set (first 5 rows, selected columns):")
    sample_cols = ['Close', 'Close_lag_1', 'Close_ma_7', 'RSI_14', 'MACD', 
                   'Gold_Oil_ratio', 'day_of_week', 'month']
    available_cols = [col for col in sample_cols if col in df_complete.columns]
    print(df_complete[available_cols].head())
    print()
    
    # Show statistics
    print("Feature Statistics:")
    print(f"  Total features: {len(df_complete.columns)}")
    print(f"  No missing values: {df_complete.isna().sum().sum() == 0}")
    print(f"  Date range: {df_complete.index.min()} to {df_complete.index.max()}")
    print()
    
    # Demonstrate convenience function
    print("Step 9: Testing convenience function engineer_features()...")
    df_quick = engineer_features(df)
    print(f"Quick feature engineering: {df_quick.shape}")
    print(f"Matches manual approach: {df_quick.shape == df_complete.shape}")
    print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("The feature engineering module successfully created:")
    print("  ✓ Lag features (1, 7, 14, 30 days)")
    print("  ✓ Rolling statistics (mean and std)")
    print("  ✓ Technical indicators (RSI, MACD, Bollinger Bands)")
    print("  ✓ Interaction features (Gold/Oil, Gold/DXY, Gold/Treasury)")
    print("  ✓ Temporal features (day, month, quarter, year, etc.)")
    print()
    print("All features are ready for model training!")


if __name__ == '__main__':
    main()
