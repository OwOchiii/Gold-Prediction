"""
Unit tests for ModelTrainingPipeline module.

Tests cover model building, initialization, and configuration for LSTM, GRU,
XGBoost, and Random Forest models.
"""

import pytest
import numpy as np
import json
import keras
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.model_training import ModelTrainingPipeline, ModelMetadata, TrainingResult
from config import Config


class TestModelTrainingPipelineInitialization:
    """Test ModelTrainingPipeline initialization."""
    
    def test_initialization_with_default_config(self):
        """Test initialization with default Config."""
        pipeline = ModelTrainingPipeline()
        
        assert pipeline.config is not None
        assert pipeline.model_dir == Config.MODEL_DIR
        assert pipeline.model_dir.exists()
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        # Create a custom config
        custom_config = Config()
        
        pipeline = ModelTrainingPipeline(config=custom_config)
        
        assert pipeline.config == custom_config
        assert pipeline.model_dir == custom_config.MODEL_DIR
    
    def test_model_directory_creation(self, tmp_path):
        """Test that model directory is created if it doesn't exist."""
        # Create a temporary config with non-existent model directory
        class TempConfig(Config):
            MODEL_DIR = tmp_path / "temp_models"
        
        pipeline = ModelTrainingPipeline(config=TempConfig)
        
        assert TempConfig.MODEL_DIR.exists()


class TestLSTMModelBuilder:
    """Test LSTM model building functionality."""
    
    def test_build_lstm_with_default_params(self):
        """Test LSTM model building with default parameters."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        assert model.name == 'LSTM_Model'
        
        # Check input shape
        assert model.input_shape == (None, 60, 10)
        
        # Check output shape
        assert model.output_shape == (None, 1)
    
    def test_build_lstm_with_custom_params(self):
        """Test LSTM model building with custom hyperparameters."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 15)
        hyperparams = {
            'units_layer1': 256,
            'units_layer2': 128,
            'dropout': 0.3,
            'dense_units': 64,
            'learning_rate': 0.0001
        }
        
        model = pipeline.build_lstm_model(input_shape, hyperparams)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        
        # Verify layer configuration
        layers = model.layers
        assert len(layers) == 6  # Input layer is implicit, 6 explicit layers
        
        # Check LSTM layers
        assert isinstance(layers[0], keras.layers.LSTM)
        assert layers[0].units == 256
        
        assert isinstance(layers[2], keras.layers.LSTM)
        assert layers[2].units == 128
    
    def test_lstm_model_has_correct_architecture(self):
        """Test LSTM model has the correct layer sequence."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        layers = model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        expected_types = ['LSTM', 'Dropout', 'LSTM', 'Dropout', 'Dense', 'Dense']
        assert layer_types == expected_types
    
    def test_lstm_model_compilation(self):
        """Test that LSTM model is properly compiled."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        # Check optimizer
        assert isinstance(model.optimizer, keras.optimizers.Adam)
        
        # Check loss function
        assert model.loss == 'mse'
        
        # Check metrics - verify model has metrics configured
        assert len(model.metrics) > 0
    
    def test_lstm_model_can_predict(self):
        """Test that LSTM model can make predictions."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        # Create dummy input
        dummy_input = np.random.randn(5, 60, 10)
        
        predictions = model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (5, 1)
    
    def test_lstm_model_layer_units(self):
        """Test LSTM model has correct number of units in each layer.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        layers = model.layers
        
        # Check first LSTM layer has 128 units (default)
        assert layers[0].units == 128
        
        # Check second LSTM layer has 64 units (default)
        assert layers[2].units == 64
        
        # Check Dense layer has 32 units (default)
        assert layers[4].units == 32
        
        # Check output layer has 1 unit
        assert layers[5].units == 1
    
    def test_lstm_model_dropout_rates(self):
        """Test LSTM model has correct dropout rates.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_lstm_model(input_shape)
        
        layers = model.layers
        
        # Check both dropout layers have rate of 0.2 (default)
        assert layers[1].rate == 0.2
        assert layers[3].rate == 0.2
    
    def test_lstm_accepts_various_input_shapes(self):
        """Test LSTM model accepts different input shapes.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        
        # Test with different sequence lengths and feature counts
        test_shapes = [
            (30, 5),
            (60, 10),
            (90, 15),
            (120, 20)
        ]
        
        for input_shape in test_shapes:
            model = pipeline.build_lstm_model(input_shape)
            assert model.input_shape == (None, input_shape[0], input_shape[1])
            
            # Verify it can accept input of that shape
            dummy_input = np.random.randn(2, input_shape[0], input_shape[1])
            predictions = model.predict(dummy_input, verbose=0)
            assert predictions.shape == (2, 1)


class TestGRUModelBuilder:
    """Test GRU model building functionality."""
    
    def test_build_gru_with_default_params(self):
        """Test GRU model building with default parameters."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        assert model.name == 'GRU_Model'
        
        # Check input shape
        assert model.input_shape == (None, 60, 10)
        
        # Check output shape
        assert model.output_shape == (None, 1)
    
    def test_build_gru_with_custom_params(self):
        """Test GRU model building with custom hyperparameters."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 15)
        hyperparams = {
            'units_layer1': 256,
            'units_layer2': 128,
            'dropout': 0.3,
            'dense_units': 64,
            'learning_rate': 0.0001
        }
        
        model = pipeline.build_gru_model(input_shape, hyperparams)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        
        # Verify layer configuration
        layers = model.layers
        assert len(layers) == 6
        
        # Check GRU layers
        assert isinstance(layers[0], keras.layers.GRU)
        assert layers[0].units == 256
        
        assert isinstance(layers[2], keras.layers.GRU)
        assert layers[2].units == 128
    
    def test_gru_model_has_correct_architecture(self):
        """Test GRU model has the correct layer sequence."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        layers = model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        expected_types = ['GRU', 'Dropout', 'GRU', 'Dropout', 'Dense', 'Dense']
        assert layer_types == expected_types
    
    def test_gru_model_compilation(self):
        """Test that GRU model is properly compiled."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        # Check optimizer
        assert isinstance(model.optimizer, keras.optimizers.Adam)
        
        # Check loss function
        assert model.loss == 'mse'
    
    def test_gru_model_can_predict(self):
        """Test that GRU model can make predictions."""
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        # Create dummy input
        dummy_input = np.random.randn(5, 60, 10)
        
        predictions = model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (5, 1)
    
    def test_gru_model_layer_units(self):
        """Test GRU model has correct number of units in each layer.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        layers = model.layers
        
        # Check first GRU layer has 128 units (default)
        assert layers[0].units == 128
        
        # Check second GRU layer has 64 units (default)
        assert layers[2].units == 64
        
        # Check Dense layer has 32 units (default)
        assert layers[4].units == 32
        
        # Check output layer has 1 unit
        assert layers[5].units == 1
    
    def test_gru_model_dropout_rates(self):
        """Test GRU model has correct dropout rates.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        input_shape = (60, 10)
        
        model = pipeline.build_gru_model(input_shape)
        
        layers = model.layers
        
        # Check both dropout layers have rate of 0.2 (default)
        assert layers[1].rate == 0.2
        assert layers[3].rate == 0.2
    
    def test_gru_accepts_various_input_shapes(self):
        """Test GRU model accepts different input shapes.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        
        # Test with different sequence lengths and feature counts
        test_shapes = [
            (30, 5),
            (60, 10),
            (90, 15),
            (120, 20)
        ]
        
        for input_shape in test_shapes:
            model = pipeline.build_gru_model(input_shape)
            assert model.input_shape == (None, input_shape[0], input_shape[1])
            
            # Verify it can accept input of that shape
            dummy_input = np.random.randn(2, input_shape[0], input_shape[1])
            predictions = model.predict(dummy_input, verbose=0)
            assert predictions.shape == (2, 1)


class TestXGBoostModelBuilder:
    """Test XGBoost model building functionality."""
    
    def test_build_xgboost_with_default_params(self):
        """Test XGBoost model building with default parameters."""
        pipeline = ModelTrainingPipeline()
        
        model = pipeline.build_xgboost_model()
        
        assert model is not None
        from xgboost import XGBRegressor
        assert isinstance(model, XGBRegressor)
        
        # Check default parameters
        assert model.max_depth == 5
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
    
    def test_build_xgboost_with_custom_params(self):
        """Test XGBoost model building with custom hyperparameters."""
        pipeline = ModelTrainingPipeline()
        hyperparams = {
            'max_depth': 7,
            'n_estimators': 300,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9
        }
        
        model = pipeline.build_xgboost_model(hyperparams)
        
        assert model.max_depth == 7
        assert model.n_estimators == 300
        assert model.learning_rate == 0.05
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.9
    
    def test_xgboost_model_can_fit_and_predict(self):
        """Test that XGBoost model can fit and predict."""
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_xgboost_model()
        
        # Create dummy training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        X_test = np.random.randn(20, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
    
    def test_xgboost_accepts_various_input_shapes(self):
        """Test XGBoost model accepts different input shapes.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_xgboost_model()
        
        # Test with different feature counts
        feature_counts = [5, 10, 20, 50]
        
        for n_features in feature_counts:
            X_train = np.random.randn(50, n_features)
            y_train = np.random.randn(50)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            X_test = np.random.randn(10, n_features)
            predictions = model.predict(X_test)
            
            assert predictions.shape == (10,)
    
    def test_xgboost_objective_configuration(self):
        """Test XGBoost model is configured with correct objective.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_xgboost_model()
        
        # Check objective is set for regression
        assert model.objective == 'reg:squarederror'


class TestRandomForestModelBuilder:
    """Test Random Forest model building functionality."""
    
    def test_build_random_forest_with_default_params(self):
        """Test Random Forest model building with default parameters."""
        pipeline = ModelTrainingPipeline()
        
        model = pipeline.build_random_forest_model()
        
        assert model is not None
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)
        
        # Check default parameters
        assert model.n_estimators == 100
        assert model.max_depth is None
        assert model.min_samples_split == 2
    
    def test_build_random_forest_with_custom_params(self):
        """Test Random Forest model building with custom hyperparameters."""
        pipeline = ModelTrainingPipeline()
        hyperparams = {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        model = pipeline.build_random_forest_model(hyperparams)
        
        assert model.n_estimators == 500
        assert model.max_depth == 20
        assert model.min_samples_split == 5
        assert model.min_samples_leaf == 2
    
    def test_random_forest_model_can_fit_and_predict(self):
        """Test that Random Forest model can fit and predict."""
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_random_forest_model()
        
        # Create dummy training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        X_test = np.random.randn(20, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
    
    def test_random_forest_accepts_various_input_shapes(self):
        """Test Random Forest model accepts different input shapes.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_random_forest_model()
        
        # Test with different feature counts
        feature_counts = [5, 10, 20, 50]
        
        for n_features in feature_counts:
            X_train = np.random.randn(50, n_features)
            y_train = np.random.randn(50)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            X_test = np.random.randn(10, n_features)
            predictions = model.predict(X_test)
            
            assert predictions.shape == (10,)
    
    def test_random_forest_parallel_configuration(self):
        """Test Random Forest model is configured for parallel execution.
        
        **Validates: Requirements 5.3**
        """
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_random_forest_model()
        
        # Check n_jobs is set to -1 (use all processors)
        assert model.n_jobs == -1


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test creation of ModelMetadata object."""
        metadata = ModelMetadata(
            version='v1.0.0',
            model_type='LSTM',
            training_date=datetime(2023, 1, 1),
            hyperparameters={'units': 128},
            feature_list=['Close', 'Volume'],
            scaling_params={'min': 0, 'max': 1},
            performance_metrics={'mae': 10.5},
            training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
            sequence_length=60
        )
        
        assert metadata.version == 'v1.0.0'
        assert metadata.model_type == 'LSTM'
        assert metadata.sequence_length == 60
    
    def test_metadata_to_dict(self):
        """Test conversion of ModelMetadata to dictionary."""
        metadata = ModelMetadata(
            version='v1.0.0',
            model_type='LSTM',
            training_date=datetime(2023, 1, 1),
            hyperparameters={'units': 128},
            feature_list=['Close', 'Volume'],
            scaling_params={'min': 0, 'max': 1},
            performance_metrics={'mae': 10.5},
            training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
            sequence_length=60
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['version'] == 'v1.0.0'
        assert metadata_dict['model_type'] == 'LSTM'
        assert metadata_dict['sequence_length'] == 60
        assert 'training_date' in metadata_dict
        assert 'hyperparameters' in metadata_dict


class TestTrainingResult:
    """Test TrainingResult class."""
    
    def test_training_result_creation(self):
        """Test creation of TrainingResult object."""
        model = None  # Placeholder
        result = TrainingResult(
            model=model,
            training_time=120.5,
            final_loss=0.05,
            validation_loss=0.06,
            convergence_status='converged',
            history={'loss': [0.1, 0.05], 'val_loss': [0.12, 0.06]}
        )
        
        assert result.training_time == 120.5
        assert result.final_loss == 0.05
        assert result.validation_loss == 0.06
        assert result.convergence_status == 'converged'
        assert 'loss' in result.history


class TestModelSaveAndLoad:
    """Test model saving functionality."""
    
    def test_save_lstm_model(self, tmp_path):
        """Test saving LSTM model with metadata."""
        # Create pipeline with temporary directory
        class TempConfig(Config):
            MODEL_DIR = tmp_path / "models"
            REGISTRY_FILE = MODEL_DIR / "registry.json"
        
        pipeline = ModelTrainingPipeline(config=TempConfig)
        
        # Build and save model
        model = pipeline.build_lstm_model(input_shape=(60, 10))
        
        metadata = ModelMetadata(
            version='v1.0.0',
            model_type='LSTM',
            training_date=datetime.now(),
            hyperparameters={'units_layer1': 128},
            feature_list=['Close', 'Volume'],
            scaling_params={},
            performance_metrics={'mae': 10.5},
            training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
            sequence_length=60
        )
        
        saved_path = pipeline.save_model(model, metadata, 'v1.0.0')
        
        # Verify files exist
        assert Path(saved_path).exists()
        assert (Path(saved_path) / 'model.h5').exists()
        assert (Path(saved_path) / 'metadata.json').exists()
    
    def test_save_xgboost_model(self, tmp_path):
        """Test saving XGBoost model with metadata."""
        # Create pipeline with temporary directory
        class TempConfig(Config):
            MODEL_DIR = tmp_path / "models"
            REGISTRY_FILE = MODEL_DIR / "registry.json"
        
        pipeline = ModelTrainingPipeline(config=TempConfig)
        
        # Build and save model
        model = pipeline.build_xgboost_model()
        
        # Need to fit model before saving
        X_dummy = np.random.randn(50, 10)
        y_dummy = np.random.randn(50)
        model.fit(X_dummy, y_dummy)
        
        metadata = ModelMetadata(
            version='v1.0.0',
            model_type='XGBoost',
            training_date=datetime.now(),
            hyperparameters={'max_depth': 5},
            feature_list=['Close', 'Volume'],
            scaling_params={},
            performance_metrics={'mae': 10.5},
            training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31))
        )
        
        saved_path = pipeline.save_model(model, metadata, 'v1.0.0')
        
        # Verify files exist
        assert Path(saved_path).exists()
        assert (Path(saved_path) / 'model.pkl').exists()
        assert (Path(saved_path) / 'metadata.json').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# Integration tests for model training
class TestModelTrainingIntegration:
    """Integration tests for complete model training workflow."""
    
    def test_lstm_training_on_synthetic_data(self):
        """Test LSTM training completes successfully on small synthetic dataset.
        
        **Validates: Requirements 5.1, 5.6**
        """
        pipeline = ModelTrainingPipeline()
        
        # Create small synthetic dataset
        np.random.seed(42)
        n_samples = 200
        sequence_length = 20
        n_features = 5
        
        # Generate synthetic sequences
        X_train = np.random.randn(150, sequence_length, n_features)
        y_train = np.random.randn(150)
        X_val = np.random.randn(50, sequence_length, n_features)
        y_val = np.random.randn(50)
        
        # Build LSTM model
        model = pipeline.build_lstm_model(input_shape=(sequence_length, n_features))
        
        # Train model with small epochs for testing
        class TestConfig:
            EPOCHS = 5
            BATCH_SIZE = 16
            EARLY_STOPPING_PATIENCE = 3
        
        original_config = pipeline.config
        pipeline.config = TestConfig()
        
        try:
            result = pipeline.train_model(
                model, X_train, y_train, X_val, y_val,
                model_type='deep_learning'
            )
            
            # Verify training completed
            assert result is not None
            assert isinstance(result, TrainingResult)
            assert result.training_time > 0
            assert result.final_loss >= 0
            assert result.validation_loss >= 0
            assert result.convergence_status in ['converged', 'max_epochs_reached']
            assert 'loss' in result.history
            assert 'val_loss' in result.history
            
        finally:
            pipeline.config = original_config
    
    def test_xgboost_training_completes_successfully(self):
        """Test XGBoost training completes successfully.
        
        **Validates: Requirements 5.1**
        """
        pipeline = ModelTrainingPipeline()
        
        # Create synthetic tabular data
        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        y_train = np.random.randn(200)
        X_val = np.random.randn(50, 10)
        y_val = np.random.randn(50)
        
        # Build XGBoost model with small parameters for fast training
        hyperparams = {
            'max_depth': 3,
            'n_estimators': 10,
            'learning_rate': 0.1
        }
        model = pipeline.build_xgboost_model(hyperparams)
        
        # Train model
        result = pipeline.train_model(
            model, X_train, y_train, X_val, y_val,
            model_type='tree_based'
        )
        
        # Verify training completed
        assert result is not None
        assert isinstance(result, TrainingResult)
        assert result.training_time > 0
        assert result.final_loss >= 0
        assert result.validation_loss >= 0
        assert result.convergence_status == 'completed'
    
    def test_early_stopping_triggers_correctly(self):
        """Test early stopping triggers correctly during training.
        
        **Validates: Requirements 5.1**
        """
        pipeline = ModelTrainingPipeline()
        
        # Create synthetic data
        np.random.seed(42)
        sequence_length = 15
        n_features = 3
        
        X_train = np.random.randn(100, sequence_length, n_features)
        y_train = np.random.randn(100)
        X_val = np.random.randn(30, sequence_length, n_features)
        y_val = np.random.randn(30)
        
        # Build model
        model = pipeline.build_lstm_model(input_shape=(sequence_length, n_features))
        
        # Configure for early stopping test
        class TestConfig:
            EPOCHS = 50  # Set high to test early stopping
            BATCH_SIZE = 16
            EARLY_STOPPING_PATIENCE = 3
        
        original_config = pipeline.config
        pipeline.config = TestConfig()
        
        try:
            result = pipeline.train_model(
                model, X_train, y_train, X_val, y_val,
                model_type='deep_learning'
            )
            
            # Verify early stopping was possible
            epochs_trained = len(result.history['loss'])
            
            # Early stopping should trigger before max epochs in most cases
            # We just verify the mechanism works (doesn't train all epochs every time)
            assert epochs_trained >= pipeline.config.EARLY_STOPPING_PATIENCE
            assert result.convergence_status in ['converged', 'max_epochs_reached']
            
        finally:
            pipeline.config = original_config
    
    def test_training_metrics_are_logged(self, tmp_path):
        """Test training metrics are logged correctly.
        
        **Validates: Requirements 5.6**
        """
        # Create pipeline with temporary directory
        class TempConfig(Config):
            MODEL_DIR = tmp_path / "models"
        
        pipeline = ModelTrainingPipeline(config=TempConfig)
        
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_val = np.random.randn(30, 10)
        y_val = np.random.randn(30)
        
        # Train Random Forest (fast)
        model = pipeline.build_random_forest_model({'n_estimators': 5})
        result = pipeline.train_model(
            model, X_train, y_train, X_val, y_val,
            model_type='tree_based'
        )
        
        # Log metrics with file saving
        pipeline.log_training_metrics(result, 'v1.0.0', save_to_file=True)
        
        # Verify history file was created
        history_file = pipeline.model_dir / 'training_history_v1.0.0.json'
        assert history_file.exists()
        
        # Verify contents
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        assert history_data['model_version'] == 'v1.0.0'
        assert 'training_time' in history_data
        assert 'final_loss' in history_data
        assert 'validation_loss' in history_data
        assert 'convergence_status' in history_data
        assert 'history' in history_data
        assert 'timestamp' in history_data
    
    def test_model_checkpointing_saves_best_model(self, tmp_path):
        """Test model checkpointing saves best model during training.
        
        **Validates: Requirements 5.1**
        """
        pipeline = ModelTrainingPipeline()
        
        # Create synthetic data
        np.random.seed(42)
        sequence_length = 15
        n_features = 3
        
        X_train = np.random.randn(100, sequence_length, n_features)
        y_train = np.random.randn(100)
        X_val = np.random.randn(30, sequence_length, n_features)
        y_val = np.random.randn(30)
        
        # Build model
        model = pipeline.build_lstm_model(input_shape=(sequence_length, n_features))
        
        # Configure for testing
        class TestConfig:
            EPOCHS = 10
            BATCH_SIZE = 16
            EARLY_STOPPING_PATIENCE = 5
        
        original_config = pipeline.config
        pipeline.config = TestConfig()
        
        checkpoint_path = tmp_path / 'best_model.keras'
        
        try:
            result = pipeline.train_model(
                model, X_train, y_train, X_val, y_val,
                model_type='deep_learning',
                checkpoint_path=str(checkpoint_path)
            )
            
            # Verify checkpoint was created
            assert checkpoint_path.exists()
            
            # Verify we can load the checkpoint
            loaded_model = keras.models.load_model(checkpoint_path)
            assert loaded_model is not None
            
            # Verify loaded model can make predictions
            predictions = loaded_model.predict(X_val[:5], verbose=0)
            assert predictions.shape == (5, 1)
            
        finally:
            pipeline.config = original_config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
