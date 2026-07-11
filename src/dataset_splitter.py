"""
Dataset Splitting and Preparation Module

This module provides functionality to split datasets chronologically for time series
modeling and prepare sequences for LSTM/GRU models.

Classes:
    - DatasetSplitter: Main class for dataset splitting and sequence preparation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from src.logger import get_logger
from config import Config

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Handles dataset splitting and preparation for model training.
    
    This class provides methods for chronological splitting, split integrity
    verification, feature-target separation, and sequence creation for
    LSTM/GRU models.
    """
    
    def __init__(self, 
                 train_ratio: float = None,
                 val_ratio: float = None,
                 test_ratio: float = None,
                 sequence_length: int = None,
                 min_records_per_split: int = None):
        """
        Initialize DatasetSplitter with configuration parameters.
        
        Args:
            train_ratio: Training set ratio (default: from Config)
            val_ratio: Validation set ratio (default: from Config)
            test_ratio: Test set ratio (default: from Config)
            sequence_length: Sequence length for LSTM/GRU (default: from Config)
            min_records_per_split: Minimum records per subset (default: from Config)
        """
        self.train_ratio = train_ratio if train_ratio is not None else Config.TRAIN_RATIO
        self.val_ratio = val_ratio if val_ratio is not None else Config.VAL_RATIO
        self.test_ratio = test_ratio if test_ratio is not None else Config.TEST_RATIO
        self.sequence_length = sequence_length if sequence_length is not None else Config.SEQUENCE_LENGTH
        self.min_records_per_split = min_records_per_split if min_records_per_split is not None else Config.MIN_RECORDS_PER_SPLIT
        
        # Validate ratios sum to 1.0
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        logger.info("DatasetSplitter initialized")
        logger.info(f"Split ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Minimum records per split: {self.min_records_per_split}")
    
    def split_dataset(self, 
                     df: pd.DataFrame,
                     train_ratio: Optional[float] = None,
                     val_ratio: Optional[float] = None,
                     test_ratio: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset chronologically into train, validation, and test sets.
        
        Performs chronological splitting to prevent look-ahead bias in time series data.
        No shuffling is performed to maintain temporal order.
        
        Args:
            df: DataFrame to split (must have datetime index or Date column)
            train_ratio: Training set ratio (optional, uses instance default)
            val_ratio: Validation set ratio (optional, uses instance default)
            test_ratio: Test set ratio (optional, uses instance default)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Raises:
            ValueError: If ratios don't sum to 1.0 or dataset is too small
            
        Requirements: 4.1, 4.2
        """
        # Use provided ratios or fall back to instance defaults
        train_r = train_ratio if train_ratio is not None else self.train_ratio
        val_r = val_ratio if val_ratio is not None else self.val_ratio
        test_r = test_ratio if test_ratio is not None else self.test_ratio
        
        # Validate ratios
        ratio_sum = train_r + val_r + test_r
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        logger.info(f"Splitting dataset with ratios - Train: {train_r}, Val: {val_r}, Test: {test_r}")
        logger.info(f"Total records: {len(df)}")
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        
        # Perform chronological split
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Train set: {len(train_df)} records")
        logger.info(f"Validation set: {len(val_df)} records")
        logger.info(f"Test set: {len(test_df)} records")
        
        # Log date ranges
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Train date range: {train_df.index.min()} to {train_df.index.max()}")
            logger.info(f"Val date range: {val_df.index.min()} to {val_df.index.max()}")
            logger.info(f"Test date range: {test_df.index.min()} to {test_df.index.max()}")
        
        return train_df, val_df, test_df
    
    def verify_split_integrity(self,
                              train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> bool:
        """
        Verify split integrity to ensure no data leakage and sufficient samples.
        
        Checks:
        1. No date overlap between splits (train < val < test chronologically)
        2. Each subset contains minimum required records
        3. No duplicate indices across splits
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame
            test_df: Test set DataFrame
        
        Returns:
            True if all integrity checks pass
            
        Raises:
            ValueError: If any integrity check fails
            
        Requirements: 4.3
        """
        logger.info("Verifying split integrity...")
        
        # Check 1: Minimum sample size
        if len(train_df) < self.min_records_per_split:
            raise ValueError(f"Training set has {len(train_df)} records, minimum required: {self.min_records_per_split}")
        
        if len(val_df) < self.min_records_per_split:
            raise ValueError(f"Validation set has {len(val_df)} records, minimum required: {self.min_records_per_split}")
        
        if len(test_df) < self.min_records_per_split:
            raise ValueError(f"Test set has {len(test_df)} records, minimum required: {self.min_records_per_split}")
        
        logger.info(f"✓ All subsets meet minimum size requirement ({self.min_records_per_split} records)")
        
        # Check 2: No date overlap (chronological order)
        if isinstance(train_df.index, pd.DatetimeIndex):
            train_max = train_df.index.max()
            val_min = val_df.index.min()
            val_max = val_df.index.max()
            test_min = test_df.index.min()
            
            if train_max >= val_min:
                raise ValueError(f"Date overlap detected: Train max ({train_max}) >= Val min ({val_min})")
            
            if val_max >= test_min:
                raise ValueError(f"Date overlap detected: Val max ({val_max}) >= Test min ({test_min})")
            
            logger.info("✓ No date overlap detected between splits")
            logger.info(f"  Train: {train_df.index.min()} to {train_max}")
            logger.info(f"  Val:   {val_min} to {val_max}")
            logger.info(f"  Test:  {test_min} to {test_df.index.max()}")
        else:
            # For non-datetime indices, check positional ordering
            logger.warning("Non-datetime index detected, skipping chronological order check")
        
        # Check 3: No duplicate indices across splits
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices
        
        if train_val_overlap:
            raise ValueError(f"Found {len(train_val_overlap)} duplicate indices between train and val sets")
        
        if train_test_overlap:
            raise ValueError(f"Found {len(train_test_overlap)} duplicate indices between train and test sets")
        
        if val_test_overlap:
            raise ValueError(f"Found {len(val_test_overlap)} duplicate indices between val and test sets")
        
        logger.info("✓ No duplicate indices found across splits")
        
        logger.info("Split integrity verification passed ✓")
        return True
    
    def prepare_feature_target_split(self,
                                    df: pd.DataFrame,
                                    target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate features (X) from target (y) and return as numpy arrays.
        
        Creates feature matrix X containing all columns except the target,
        and target vector y containing only the target column.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column (default: from Config)
        
        Returns:
            Tuple of (X, y) as numpy arrays
            
        Raises:
            ValueError: If target column is missing
            
        Requirements: 4.4
        """
        target_col = target_column if target_column is not None else Config.TARGET_COLUMN
        
        logger.info(f"Preparing feature-target split with target column: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        # Separate features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        logger.info(f"Feature matrix X shape: {X.shape}")
        logger.info(f"Target vector y shape: {y.shape}")
        logger.info(f"Number of features: {X.shape[1]}")
        
        return X, y
    
    def create_sequences(self,
                        data: np.ndarray,
                        target: np.ndarray,
                        sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM/GRU models.
        
        Transforms time series data into sequences of fixed length. Each sequence
        contains 'sequence_length' time steps and is used to predict the next value.
        
        Example with sequence_length=3:
            Input: [1, 2, 3, 4, 5]
            Sequences: [[1, 2, 3], [2, 3, 4]]
            Targets: [4, 5]
        
        Args:
            data: Feature matrix (n_samples, n_features)
            target: Target vector (n_samples,)
            sequence_length: Length of each sequence (default: uses instance setting)
        
        Returns:
            Tuple of (X_sequences, y_targets)
            - X_sequences shape: (n_sequences, sequence_length, n_features)
            - y_targets shape: (n_sequences,)
            
        Raises:
            ValueError: If data is too short to create any sequences
            
        Requirements: 4.4
        """
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        
        logger.info(f"Creating sequences with length {seq_len}")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input target shape: {target.shape}")
        
        # Validate inputs
        if len(data) != len(target):
            raise ValueError(f"Data and target must have same length. Got data: {len(data)}, target: {len(target)}")
        
        if len(data) <= seq_len:
            raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({seq_len})")
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(data) - seq_len):
            # Extract sequence of length seq_len
            sequence = data[i:i + seq_len]
            # Target is the next value after the sequence
            target_value = target[i + seq_len]
            
            X_sequences.append(sequence)
            y_targets.append(target_value)
        
        # Convert to numpy arrays
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        logger.info(f"Created {len(X_sequences)} sequences")
        logger.info(f"X_sequences shape: {X_sequences.shape}")
        logger.info(f"y_targets shape: {y_targets.shape}")
        
        # Verify shapes
        expected_shape = (len(data) - seq_len, seq_len, data.shape[1] if data.ndim > 1 else 1)
        if data.ndim == 1:
            # If input is 1D, reshape sequences to (n_sequences, seq_len, 1)
            X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        logger.info(f"Final X_sequences shape: {X_sequences.shape}")
        
        return X_sequences, y_targets
