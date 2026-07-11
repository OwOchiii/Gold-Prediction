"""
Unit tests for DatasetSplitter class.

Tests cover chronological splitting, integrity verification,
feature-target separation, and sequence creation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.dataset_splitter import DatasetSplitter
from config import Config


class TestDatasetSplitterInitialization:
    """Test DatasetSplitter initialization and configuration."""
    
    def test_initialization_with_defaults(self):
        """Test initialization uses Config defaults."""
        splitter = DatasetSplitter()
        
        assert splitter.train_ratio == Config.TRAIN_RATIO
        assert splitter.val_ratio == Config.VAL_RATIO
        assert splitter.test_ratio == Config.TEST_RATIO
        assert splitter.sequence_length == Config.SEQUENCE_LENGTH
        assert splitter.min_records_per_split == Config.MIN_RECORDS_PER_SPLIT
    
    def test_initialization_with_custom_values(self):
        """Test initialization with custom parameters."""
        splitter = DatasetSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            sequence_length=30,
            min_records_per_split=50
        )
        
        assert splitter.train_ratio == 0.6
        assert splitter.val_ratio == 0.2
        assert splitter.test_ratio == 0.2
        assert splitter.sequence_length == 30
        assert splitter.min_records_per_split == 50
    
    def test_initialization_invalid_ratios(self):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            DatasetSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestChronologicalSplitting:
    """Test chronological dataset splitting functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with datetime index."""
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 1000),
            'Feature1': np.random.randn(1000),
            'Feature2': np.random.randn(1000)
        }, index=dates)
        return df
    
    def test_split_dataset_basic(self, sample_dataframe):
        """Test basic dataset splitting with default ratios."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split_dataset(sample_dataframe)
        
        # Check sizes match ratios
        total = len(sample_dataframe)
        assert len(train_df) == int(total * 0.7)
        assert len(val_df) == int(total * 0.15)
        assert len(test_df) == total - len(train_df) - len(val_df)
        
        # Check chronological order
        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()
    
    def test_split_dataset_custom_ratios(self, sample_dataframe):
        """Test splitting with custom ratios."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split_dataset(
            sample_dataframe,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        total = len(sample_dataframe)
        assert len(train_df) == int(total * 0.6)
        assert len(val_df) == int(total * 0.2)
        assert len(test_df) == total - len(train_df) - len(val_df)
    
    def test_split_dataset_no_overlap(self, sample_dataframe):
        """Test that splits have no overlapping indices."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split_dataset(sample_dataframe)
        
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0
    
    def test_split_dataset_preserves_columns(self, sample_dataframe):
        """Test that all splits preserve DataFrame columns."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split_dataset(sample_dataframe)
        
        assert list(train_df.columns) == list(sample_dataframe.columns)
        assert list(val_df.columns) == list(sample_dataframe.columns)
        assert list(test_df.columns) == list(sample_dataframe.columns)
    
    def test_split_dataset_invalid_ratios(self, sample_dataframe):
        """Test that invalid ratios raise ValueError."""
        splitter = DatasetSplitter()
        
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            splitter.split_dataset(
                sample_dataframe,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3
            )


class TestSplitIntegrityVerification:
    """Test split integrity verification functionality."""
    
    @pytest.fixture
    def valid_splits(self):
        """Create valid train/val/test splits."""
        dates_train = pd.date_range(start='2020-01-01', periods=200, freq='D')
        dates_val = pd.date_range(start='2020-07-20', periods=100, freq='D')
        dates_test = pd.date_range(start='2020-10-28', periods=100, freq='D')
        
        train_df = pd.DataFrame({
            'Close': np.random.randn(200).cumsum() + 100,
            'Feature1': np.random.randn(200)
        }, index=dates_train)
        
        val_df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Feature1': np.random.randn(100)
        }, index=dates_val)
        
        test_df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Feature1': np.random.randn(100)
        }, index=dates_test)
        
        return train_df, val_df, test_df
    
    def test_verify_split_integrity_valid(self, valid_splits):
        """Test verification passes for valid splits."""
        train_df, val_df, test_df = valid_splits
        splitter = DatasetSplitter()
        
        result = splitter.verify_split_integrity(train_df, val_df, test_df)
        assert result is True
    
    def test_verify_split_integrity_insufficient_train_samples(self, valid_splits):
        """Test verification fails when train set is too small."""
        train_df, val_df, test_df = valid_splits
        
        # Create small training set
        small_train = train_df.iloc[:50]
        
        splitter = DatasetSplitter(min_records_per_split=100)
        
        with pytest.raises(ValueError, match="Training set has 50 records, minimum required: 100"):
            splitter.verify_split_integrity(small_train, val_df, test_df)
    
    def test_verify_split_integrity_insufficient_val_samples(self, valid_splits):
        """Test verification fails when validation set is too small."""
        train_df, val_df, test_df = valid_splits
        
        # Create small validation set
        small_val = val_df.iloc[:80]
        
        splitter = DatasetSplitter(min_records_per_split=100)
        
        with pytest.raises(ValueError, match="Validation set has 80 records, minimum required: 100"):
            splitter.verify_split_integrity(train_df, small_val, test_df)
    
    def test_verify_split_integrity_insufficient_test_samples(self, valid_splits):
        """Test verification fails when test set is too small."""
        train_df, val_df, test_df = valid_splits
        
        # Create small test set
        small_test = test_df.iloc[:80]
        
        splitter = DatasetSplitter(min_records_per_split=100)
        
        with pytest.raises(ValueError, match="Test set has 80 records, minimum required: 100"):
            splitter.verify_split_integrity(train_df, val_df, small_test)
    
    def test_verify_split_integrity_date_overlap_train_val(self, valid_splits):
        """Test verification fails when train and val dates overlap."""
        train_df, val_df, test_df = valid_splits
        
        # Create overlapping validation set (100 records to pass size check)
        overlapping_val = train_df.iloc[-100:].copy()
        
        splitter = DatasetSplitter()
        
        with pytest.raises(ValueError, match="Date overlap detected: Train max .* >= Val min"):
            splitter.verify_split_integrity(train_df, overlapping_val, test_df)
    
    def test_verify_split_integrity_date_overlap_val_test(self, valid_splits):
        """Test verification fails when val and test dates overlap."""
        train_df, val_df, test_df = valid_splits
        
        # Create overlapping test set (100 records to pass size check)
        overlapping_test = val_df.iloc[:100].copy()
        
        splitter = DatasetSplitter()
        
        with pytest.raises(ValueError, match="Date overlap detected: Val max .* >= Test min"):
            splitter.verify_split_integrity(train_df, val_df, overlapping_test)
    
    def test_verify_split_integrity_duplicate_indices(self):
        """Test verification fails when there are duplicate indices across splits."""
        # Create splits without datetime index to bypass date overlap check
        train_df = pd.DataFrame({
            'Close': np.random.randn(200),
            'Feature1': np.random.randn(200)
        }, index=range(200))
        
        val_df = pd.DataFrame({
            'Close': np.random.randn(100),
            'Feature1': np.random.randn(100)
        }, index=range(200, 300))
        
        # Create test with duplicate indices from train
        test_df = pd.DataFrame({
            'Close': np.random.randn(100),
            'Feature1': np.random.randn(100)
        }, index=range(100, 200))  # Overlaps with train indices 100-199
        
        splitter = DatasetSplitter()
        
        with pytest.raises(ValueError, match="duplicate indices between train and test sets"):
            splitter.verify_split_integrity(train_df, val_df, test_df)


class TestFeatureTargetSeparation:
    """Test feature-target separation functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with features and target."""
        df = pd.DataFrame({
            'Close': np.array([100, 101, 102, 103, 104]),
            'Feature1': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'Feature2': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            'Feature3': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        })
        return df
    
    def test_prepare_feature_target_split_default_target(self, sample_dataframe):
        """Test feature-target separation with default target column."""
        splitter = DatasetSplitter()
        X, y = splitter.prepare_feature_target_split(sample_dataframe)
        
        # Check shapes
        assert X.shape == (5, 3)  # 5 samples, 3 features (excluding Close)
        assert y.shape == (5,)  # 5 target values
        
        # Check target values
        np.testing.assert_array_equal(y, np.array([100, 101, 102, 103, 104]))
        
        # Check that Close is not in features
        assert sample_dataframe['Close'].values[0] not in X[0]
    
    def test_prepare_feature_target_split_custom_target(self, sample_dataframe):
        """Test feature-target separation with custom target column."""
        splitter = DatasetSplitter()
        X, y = splitter.prepare_feature_target_split(sample_dataframe, target_column='Feature1')
        
        # Check shapes
        assert X.shape == (5, 3)  # 5 samples, 3 features (excluding Feature1)
        assert y.shape == (5,)
        
        # Check target values
        np.testing.assert_array_equal(y, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    
    def test_prepare_feature_target_split_missing_target(self, sample_dataframe):
        """Test that missing target column raises ValueError."""
        splitter = DatasetSplitter()
        
        with pytest.raises(ValueError, match="Target column 'NonExistent' not found"):
            splitter.prepare_feature_target_split(sample_dataframe, target_column='NonExistent')
    
    def test_prepare_feature_target_split_returns_numpy_arrays(self, sample_dataframe):
        """Test that returned values are numpy arrays."""
        splitter = DatasetSplitter()
        X, y = splitter.prepare_feature_target_split(sample_dataframe)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestSequenceCreation:
    """Test sequence creation for LSTM/GRU models."""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation with 2D data."""
        splitter = DatasetSplitter(sequence_length=3)
        
        # Create simple test data
        data = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50]
        ])
        target = np.array([100, 200, 300, 400, 500])
        
        X_seq, y_seq = splitter.create_sequences(data, target, sequence_length=3)
        
        # Check shapes
        assert X_seq.shape == (2, 3, 2)  # 2 sequences, length 3, 2 features
        assert y_seq.shape == (2,)  # 2 targets
        
        # Check first sequence
        expected_first_seq = np.array([[1, 10], [2, 20], [3, 30]])
        np.testing.assert_array_equal(X_seq[0], expected_first_seq)
        assert y_seq[0] == 400
        
        # Check second sequence
        expected_second_seq = np.array([[2, 20], [3, 30], [4, 40]])
        np.testing.assert_array_equal(X_seq[1], expected_second_seq)
        assert y_seq[1] == 500
    
    def test_create_sequences_default_length(self):
        """Test sequence creation with default sequence length from config."""
        splitter = DatasetSplitter()  # Uses Config.SEQUENCE_LENGTH (60)
        
        # Create data with enough samples
        n_samples = 100
        data = np.random.randn(n_samples, 5)
        target = np.random.randn(n_samples)
        
        X_seq, y_seq = splitter.create_sequences(data, target)
        
        # Should create 40 sequences (100 - 60 = 40)
        assert X_seq.shape == (40, 60, 5)
        assert y_seq.shape == (40,)
    
    def test_create_sequences_custom_length(self):
        """Test sequence creation with custom sequence length."""
        splitter = DatasetSplitter()
        
        data = np.random.randn(50, 3)
        target = np.random.randn(50)
        
        X_seq, y_seq = splitter.create_sequences(data, target, sequence_length=10)
        
        # Should create 40 sequences (50 - 10 = 40)
        assert X_seq.shape == (40, 10, 3)
        assert y_seq.shape == (40,)
    
    def test_create_sequences_1d_data(self):
        """Test sequence creation with 1D data."""
        splitter = DatasetSplitter(sequence_length=3)
        
        data = np.array([1, 2, 3, 4, 5])
        target = np.array([10, 20, 30, 40, 50])
        
        X_seq, y_seq = splitter.create_sequences(data, target)
        
        # Should reshape 1D to (n_sequences, seq_length, 1)
        assert X_seq.shape == (2, 3, 1)
        assert y_seq.shape == (2,)
        
        # Check values
        np.testing.assert_array_equal(X_seq[0].flatten(), np.array([1, 2, 3]))
        assert y_seq[0] == 40
    
    def test_create_sequences_sliding_window(self):
        """Test that sequences correctly implement sliding window."""
        splitter = DatasetSplitter(sequence_length=3)
        
        data = np.array([[i] for i in range(1, 6)])  # [[1], [2], [3], [4], [5]]
        target = np.array([10, 20, 30, 40, 50])
        
        X_seq, y_seq = splitter.create_sequences(data, target)
        
        # First sequence: [1, 2, 3] -> target 40
        # Second sequence: [2, 3, 4] -> target 50
        assert X_seq[0, 0, 0] == 1
        assert X_seq[0, 1, 0] == 2
        assert X_seq[0, 2, 0] == 3
        assert y_seq[0] == 40
        
        assert X_seq[1, 0, 0] == 2
        assert X_seq[1, 1, 0] == 3
        assert X_seq[1, 2, 0] == 4
        assert y_seq[1] == 50
    
    def test_create_sequences_mismatched_lengths(self):
        """Test that mismatched data and target lengths raise ValueError."""
        splitter = DatasetSplitter()
        
        data = np.random.randn(50, 3)
        target = np.random.randn(40)  # Mismatched length
        
        with pytest.raises(ValueError, match="Data and target must have same length"):
            splitter.create_sequences(data, target)
    
    def test_create_sequences_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        splitter = DatasetSplitter(sequence_length=10)
        
        data = np.random.randn(5, 3)  # Only 5 samples, need > 10
        target = np.random.randn(5)
        
        with pytest.raises(ValueError, match="Data length .* must be greater than sequence length"):
            splitter.create_sequences(data, target)
    
    def test_create_sequences_exact_length(self):
        """Test that data exactly equal to sequence length raises ValueError."""
        splitter = DatasetSplitter(sequence_length=5)
        
        data = np.random.randn(5, 3)
        target = np.random.randn(5)
        
        with pytest.raises(ValueError, match="Data length .* must be greater than sequence length"):
            splitter.create_sequences(data, target)


class TestIntegration:
    """Integration tests for complete dataset preparation workflow."""
    
    @pytest.fixture
    def full_dataset(self):
        """Create a complete dataset for integration testing."""
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100,
            'Open': np.random.randn(1000).cumsum() + 100,
            'High': np.random.randn(1000).cumsum() + 105,
            'Low': np.random.randn(1000).cumsum() + 95,
            'Volume': np.random.randint(1000, 10000, 1000),
            'DXY': np.random.randn(1000).cumsum() + 95,
            'Oil': np.random.randn(1000).cumsum() + 70
        }, index=dates)
        return df
    
    def test_complete_workflow(self, full_dataset):
        """Test complete workflow: split -> verify -> prepare -> sequences."""
        splitter = DatasetSplitter()
        
        # Step 1: Split dataset
        train_df, val_df, test_df = splitter.split_dataset(full_dataset)
        
        # Step 2: Verify integrity
        assert splitter.verify_split_integrity(train_df, val_df, test_df) is True
        
        # Step 3: Prepare feature-target splits
        X_train, y_train = splitter.prepare_feature_target_split(train_df)
        X_val, y_val = splitter.prepare_feature_target_split(val_df)
        X_test, y_test = splitter.prepare_feature_target_split(test_df)
        
        # Verify shapes
        assert X_train.shape[0] == len(train_df)
        assert X_val.shape[0] == len(val_df)
        assert X_test.shape[0] == len(test_df)
        
        # Step 4: Create sequences for LSTM/GRU
        X_train_seq, y_train_seq = splitter.create_sequences(X_train, y_train, sequence_length=10)
        X_val_seq, y_val_seq = splitter.create_sequences(X_val, y_val, sequence_length=10)
        X_test_seq, y_test_seq = splitter.create_sequences(X_test, y_test, sequence_length=10)
        
        # Verify sequence shapes
        assert X_train_seq.ndim == 3
        assert X_train_seq.shape[1] == 10  # Sequence length
        assert X_train_seq.shape[2] == X_train.shape[1]  # Number of features
        
        # Verify all steps completed without errors
        assert X_train_seq.shape[0] > 0
        assert X_val_seq.shape[0] > 0
        assert X_test_seq.shape[0] > 0
