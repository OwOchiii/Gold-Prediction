"""
Demonstration script for data preprocessing module.

This script demonstrates the complete data preprocessing pipeline including:
- Loading data
- Handling missing values
- Normalizing features
- Aligning datasets
- Removing outliers
- Generating quality reports
"""

import pandas as pd
import numpy as np
from datetime import datetime

from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from src.logger import get_logger
from config import Config

logger = get_logger(__name__)


def main():
    """Demonstrate the data preprocessing pipeline."""
    
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE DEMONSTRATION")
    logger.info("=" * 60)
    
    # Initialize components
    ingestion_manager = DataIngestionManager()
    preprocessor = DataPreprocessor()
    
    # Step 1: Load gold OHLCV data
    logger.info("\n1. Loading gold OHLCV data...")
    try:
        gold_df = ingestion_manager.load_csv('XAU_1d_data.csv')
        logger.info(f"Loaded {len(gold_df)} records of gold price data")
        logger.info(f"Date range: {gold_df.index.min()} to {gold_df.index.max()}")
        
        # Track original count for quality report
        original_count = len(gold_df)
        preprocessor.stats['original_record_count'] = original_count
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Step 2: Handle missing values
    logger.info("\n2. Handling missing values...")
    missing_info = ingestion_manager.detect_missing_values(gold_df)
    if missing_info:
        logger.info(f"Missing values detected in {len(missing_info)} columns")
        gold_df = preprocessor.handle_missing_values(gold_df, max_gap=3)
        logger.info("Missing values handled with forward-fill")
    else:
        logger.info("No missing values detected")
    
    # Step 3: Remove outliers
    logger.info("\n3. Removing outliers...")
    gold_df_clean = preprocessor.remove_outliers(
        gold_df,
        n_std=Config.OUTLIER_STD_THRESHOLD,
        columns=['Close', 'High', 'Low', 'Open']
    )
    logger.info(f"Records after outlier removal: {len(gold_df_clean)}")
    
    # Step 4: Normalize features
    logger.info("\n4. Normalizing features...")
    gold_df_normalized, scaling_params = preprocessor.normalize_features(
        gold_df_clean,
        method=Config.NORMALIZATION_METHOD
    )
    logger.info(f"Normalized using {scaling_params['method']} method")
    logger.info(f"Example - Close price range: [{gold_df_normalized['Close'].min():.4f}, {gold_df_normalized['Close'].max():.4f}]")
    
    # Step 5: Generate quality report
    logger.info("\n5. Generating data quality report...")
    quality_report = preprocessor.generate_quality_report(
        gold_df_normalized,
        original_record_count=original_count
    )
    
    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original records: {original_count}")
    logger.info(f"Final records: {quality_report.total_records}")
    logger.info(f"Records retained: {quality_report.total_records/original_count*100:.2f}%")
    logger.info(f"Missing values handled: {sum(quality_report.missing_values_handled.values())}")
    logger.info(f"Outliers removed: {sum(quality_report.outliers_removed.values())}")
    logger.info(f"Data quality score: {quality_report.data_quality_score:.2f}/100")
    logger.info("=" * 60)
    
    # Step 6: Demonstrate denormalization
    logger.info("\n6. Demonstrating denormalization...")
    gold_df_denormalized = preprocessor.denormalize_features(
        gold_df_normalized,
        scaling_params
    )
    logger.info("Denormalization complete - data restored to original scale")
    
    # Verify denormalization accuracy
    original_sample = gold_df_clean['Close'].iloc[:5]
    denorm_sample = gold_df_denormalized['Close'].iloc[:5]
    logger.info(f"Original Close prices (first 5): {original_sample.values}")
    logger.info(f"After normalize/denormalize: {denorm_sample.values}")
    
    logger.info("\n✓ Data preprocessing pipeline demonstration complete!")


if __name__ == '__main__':
    main()
