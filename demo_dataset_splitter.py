"""
Demo script for DatasetSplitter class functionality.

This script demonstrates:
1. Loading and preparing data
2. Chronological splitting into train/val/test sets
3. Verifying split integrity
4. Feature-target separation
5. Creating sequences for LSTM/GRU models
"""

import pandas as pd
import numpy as np
from datetime import datetime

from src.dataset_splitter import DatasetSplitter
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    """Demonstrate DatasetSplitter functionality."""
    
    print("=" * 80)
    print("DATASET SPLITTER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Step 1: Create sample dataset
    print("Step 1: Creating sample dataset...")
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'Close': np.random.randn(1000).cumsum() + 1800,  # Gold price around $1800
        'Open': np.random.randn(1000).cumsum() + 1800,
        'High': np.random.randn(1000).cumsum() + 1850,
        'Low': np.random.randn(1000).cumsum() + 1750,
        'Volume': np.random.randint(50000, 200000, 1000),
        'DXY': np.random.randn(1000).cumsum() + 95,
        'Oil': np.random.randn(1000).cumsum() + 70,
        'Treasury_10Y': np.random.randn(1000) * 0.5 + 2.0
    }, index=dates)
    
    print(f"Created dataset with {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Step 2: Initialize DatasetSplitter
    print("Step 2: Initializing DatasetSplitter...")
    splitter = DatasetSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sequence_length=60,
        min_records_per_split=100
    )
    print("DatasetSplitter initialized with:")
    print(f"  - Train ratio: {splitter.train_ratio}")
    print(f"  - Validation ratio: {splitter.val_ratio}")
    print(f"  - Test ratio: {splitter.test_ratio}")
    print(f"  - Sequence length: {splitter.sequence_length}")
    print(f"  - Min records per split: {splitter.min_records_per_split}")
    print()
    
    # Step 3: Split dataset chronologically
    print("Step 3: Splitting dataset chronologically...")
    train_df, val_df, test_df = splitter.split_dataset(df)
    
    print(f"Train set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {train_df.index.min()} to {train_df.index.max()}")
    
    print(f"Validation set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {val_df.index.min()} to {val_df.index.max()}")
    
    print(f"Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {test_df.index.min()} to {test_df.index.max()}")
    print()
    
    # Step 4: Verify split integrity
    print("Step 4: Verifying split integrity...")
    try:
        integrity_ok = splitter.verify_split_integrity(train_df, val_df, test_df)
        print(f"✓ Split integrity verification: {'PASSED' if integrity_ok else 'FAILED'}")
    except ValueError as e:
        print(f"✗ Split integrity verification FAILED: {e}")
    print()
    
    # Step 5: Prepare feature-target splits
    print("Step 5: Preparing feature-target splits...")
    X_train, y_train = splitter.prepare_feature_target_split(train_df)
    X_val, y_val = splitter.prepare_feature_target_split(val_df)
    X_test, y_test = splitter.prepare_feature_target_split(test_df)
    
    print(f"Train features (X): {X_train.shape}")
    print(f"Train target (y): {y_train.shape}")
    print(f"Validation features (X): {X_val.shape}")
    print(f"Validation target (y): {y_val.shape}")
    print(f"Test features (X): {X_test.shape}")
    print(f"Test target (y): {y_test.shape}")
    print()
    
    # Step 6: Create sequences for LSTM/GRU
    print("Step 6: Creating sequences for LSTM/GRU models...")
    X_train_seq, y_train_seq = splitter.create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = splitter.create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = splitter.create_sequences(X_test, y_test)
    
    print(f"Train sequences (X): {X_train_seq.shape}")
    print(f"  - Number of sequences: {X_train_seq.shape[0]}")
    print(f"  - Sequence length: {X_train_seq.shape[1]}")
    print(f"  - Number of features: {X_train_seq.shape[2]}")
    print(f"Train targets (y): {y_train_seq.shape}")
    
    print(f"Validation sequences (X): {X_val_seq.shape}")
    print(f"Validation targets (y): {y_val_seq.shape}")
    
    print(f"Test sequences (X): {X_test_seq.shape}")
    print(f"Test targets (y): {y_test_seq.shape}")
    print()
    
    # Step 7: Show example sequence
    print("Step 7: Example sequence (first sequence from training set)...")
    print(f"Sequence shape: {X_train_seq[0].shape}")
    print(f"First 5 timesteps of first feature:")
    print(X_train_seq[0][:5, 0])
    print(f"Target value for this sequence: {y_train_seq[0]:.2f}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original dataset: {len(df)} records")
    print(f"After splitting:")
    print(f"  - Train: {len(train_df)} records → {X_train_seq.shape[0]} sequences")
    print(f"  - Val: {len(val_df)} records → {X_val_seq.shape[0]} sequences")
    print(f"  - Test: {len(test_df)} records → {X_test_seq.shape[0]} sequences")
    print()
    print("✓ Dataset splitting and preparation complete!")
    print("✓ Ready for LSTM/GRU model training")
    print("=" * 80)


if __name__ == "__main__":
    main()
