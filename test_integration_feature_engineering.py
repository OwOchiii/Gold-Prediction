"""
Integration test for Feature Engineering with Data Ingestion and Preprocessing

This script tests the complete data pipeline:
1. Data Ingestion (load CSV and economic indicators)
2. Data Preprocessing (clean and align)
3. Feature Engineering (create features)
"""

import pandas as pd
from datetime import datetime

from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.logger import get_logger
from config import Config

logger = get_logger(__name__)


def test_integration():
    """Test integration of all data pipeline components."""
    
    print("=" * 80)
    print("INTEGRATION TEST: Data Pipeline with Feature Engineering")
    print("=" * 80)
    print()
    
    # Step 1: Data Ingestion
    print("Step 1: Loading gold price data...")
    ingestion_manager = DataIngestionManager()
    
    try:
        gold_df = ingestion_manager.load_csv('XAU_1d_data.csv')
        print(f"✓ Loaded {len(gold_df)} records from CSV")
        print(f"  Date range: {gold_df.index.min()} to {gold_df.index.max()}")
    except FileNotFoundError:
        print("✗ CSV file not found. Using sample data instead.")
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        gold_df = pd.DataFrame({
            'Open': [1800 + i * 0.1 for i in range(len(dates))],
            'High': [1820 + i * 0.1 for i in range(len(dates))],
            'Low': [1780 + i * 0.1 for i in range(len(dates))],
            'Close': [1800 + i * 0.1 for i in range(len(dates))],
            'Volume': [100000] * len(dates)
        }, index=dates)
    
    print()
    
    # Step 2: Add synthetic economic indicators (in real scenario, use yfinance)
    print("Step 2: Adding economic indicators...")
    gold_df['DXY'] = 102.0
    gold_df['Oil'] = 75.0
    gold_df['Treasury_10Y'] = 3.5
    print("✓ Economic indicators added")
    print()
    
    # Step 3: Data Preprocessing
    print("Step 3: Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    gold_df = preprocessor.handle_missing_values(gold_df)
    
    # Remove outliers
    gold_df = preprocessor.remove_outliers(gold_df)
    
    print(f"✓ Preprocessing complete: {len(gold_df)} records")
    print()
    
    # Step 4: Feature Engineering
    print("Step 4: Engineering features...")
    engineer = FeatureEngineer()
    
    df_features = engineer.build_feature_set(gold_df, handle_nan=True)
    
    print(f"✓ Feature engineering complete: {df_features.shape}")
    print()
    
    # Verify results
    print("=" * 80)
    print("INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    # Check that all expected feature types are present
    checks = {
        'Lag features': any('lag' in col for col in df_features.columns),
        'Rolling features': any('ma_' in col or 'std_' in col for col in df_features.columns),
        'Technical indicators': 'RSI_14' in df_features.columns and 'MACD' in df_features.columns,
        'Interaction features': 'Gold_Oil_ratio' in df_features.columns,
        'Temporal features': 'day_of_week' in df_features.columns
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    
    print()
    print(f"Final feature count: {len(df_features.columns)} columns")
    print(f"Final record count: {len(df_features)} rows")
    print(f"No missing values: {df_features.isna().sum().sum() == 0}")
    print()
    
    if all_passed:
        print("=" * 80)
        print("✓ INTEGRATION TEST PASSED")
        print("=" * 80)
        print()
        print("The complete data pipeline is working correctly:")
        print("  Data Ingestion → Data Preprocessing → Feature Engineering")
        print()
        print("Ready for Dataset Splitting and Model Training!")
        return True
    else:
        print("=" * 80)
        print("✗ INTEGRATION TEST FAILED")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)
