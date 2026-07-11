"""
Demo script for ModelTrainingPipeline module.

This script demonstrates the usage of the ModelTrainingPipeline class
for building LSTM, GRU, XGBoost, and Random Forest models.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'  # Set backend before importing keras

import numpy as np
from datetime import datetime

from src.model_training import ModelTrainingPipeline, ModelMetadata


def demo_lstm_model():
    """Demonstrate LSTM model building."""
    print("\n" + "="*60)
    print("LSTM Model Demo")
    print("="*60)
    
    pipeline = ModelTrainingPipeline()
    
    # Build LSTM model with default parameters
    print("\n1. Building LSTM model with default parameters...")
    input_shape = (60, 15)  # 60 time steps, 15 features
    lstm_model = pipeline.build_lstm_model(input_shape)
    
    print(f"   ✓ Model created: {lstm_model.name}")
    print(f"   ✓ Input shape: {lstm_model.input_shape}")
    print(f"   ✓ Output shape: {lstm_model.output_shape}")
    print(f"   ✓ Total parameters: {lstm_model.count_params():,}")
    
    # Build LSTM model with custom hyperparameters
    print("\n2. Building LSTM model with custom hyperparameters...")
    hyperparams = {
        'units_layer1': 256,
        'units_layer2': 128,
        'dropout': 0.3,
        'dense_units': 64,
        'learning_rate': 0.0001
    }
    lstm_custom = pipeline.build_lstm_model(input_shape, hyperparams)
    
    print(f"   ✓ Model created with custom params")
    print(f"   ✓ Total parameters: {lstm_custom.count_params():,}")
    
    # Test prediction
    print("\n3. Testing LSTM prediction...")
    dummy_data = np.random.randn(10, 60, 15)
    predictions = lstm_model.predict(dummy_data, verbose=0)
    
    print(f"   ✓ Input shape: {dummy_data.shape}")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Sample predictions: {predictions[:3].flatten()}")


def demo_gru_model():
    """Demonstrate GRU model building."""
    print("\n" + "="*60)
    print("GRU Model Demo")
    print("="*60)
    
    pipeline = ModelTrainingPipeline()
    
    # Build GRU model
    print("\n1. Building GRU model...")
    input_shape = (60, 15)
    gru_model = pipeline.build_gru_model(input_shape)
    
    print(f"   ✓ Model created: {gru_model.name}")
    print(f"   ✓ Total parameters: {gru_model.count_params():,}")
    
    # Test prediction
    print("\n2. Testing GRU prediction...")
    dummy_data = np.random.randn(5, 60, 15)
    predictions = gru_model.predict(dummy_data, verbose=0)
    
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Sample predictions: {predictions[:3].flatten()}")


def demo_xgboost_model():
    """Demonstrate XGBoost model building."""
    print("\n" + "="*60)
    print("XGBoost Model Demo")
    print("="*60)
    
    pipeline = ModelTrainingPipeline()
    
    # Build XGBoost model with default parameters
    print("\n1. Building XGBoost model with default parameters...")
    xgb_model = pipeline.build_xgboost_model()
    
    print(f"   ✓ Model created: {type(xgb_model).__name__}")
    print(f"   ✓ Max depth: {xgb_model.max_depth}")
    print(f"   ✓ N estimators: {xgb_model.n_estimators}")
    print(f"   ✓ Learning rate: {xgb_model.learning_rate}")
    
    # Build XGBoost model with custom hyperparameters
    print("\n2. Building XGBoost model with custom hyperparameters...")
    hyperparams = {
        'max_depth': 7,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.9
    }
    xgb_custom = pipeline.build_xgboost_model(hyperparams)
    
    print(f"   ✓ Model created with custom params")
    print(f"   ✓ Max depth: {xgb_custom.max_depth}")
    print(f"   ✓ N estimators: {xgb_custom.n_estimators}")
    
    # Test fit and predict
    print("\n3. Testing XGBoost fit and predict...")
    X_train = np.random.randn(100, 15)
    y_train = np.random.randn(100)
    
    xgb_model.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 15)
    predictions = xgb_model.predict(X_test)
    
    print(f"   ✓ Training data shape: {X_train.shape}")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Sample predictions: {predictions[:3]}")


def demo_random_forest_model():
    """Demonstrate Random Forest model building."""
    print("\n" + "="*60)
    print("Random Forest Model Demo")
    print("="*60)
    
    pipeline = ModelTrainingPipeline()
    
    # Build Random Forest model with default parameters
    print("\n1. Building Random Forest model with default parameters...")
    rf_model = pipeline.build_random_forest_model()
    
    print(f"   ✓ Model created: {type(rf_model).__name__}")
    print(f"   ✓ N estimators: {rf_model.n_estimators}")
    print(f"   ✓ Max depth: {rf_model.max_depth}")
    print(f"   ✓ Min samples split: {rf_model.min_samples_split}")
    
    # Build Random Forest model with custom hyperparameters
    print("\n2. Building Random Forest model with custom hyperparameters...")
    hyperparams = {
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
    rf_custom = pipeline.build_random_forest_model(hyperparams)
    
    print(f"   ✓ Model created with custom params")
    print(f"   ✓ N estimators: {rf_custom.n_estimators}")
    print(f"   ✓ Max depth: {rf_custom.max_depth}")
    
    # Test fit and predict
    print("\n3. Testing Random Forest fit and predict...")
    X_train = np.random.randn(100, 15)
    y_train = np.random.randn(100)
    
    rf_model.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 15)
    predictions = rf_model.predict(X_test)
    
    print(f"   ✓ Training data shape: {X_train.shape}")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Sample predictions: {predictions[:3]}")


def demo_model_metadata():
    """Demonstrate ModelMetadata usage."""
    print("\n" + "="*60)
    print("Model Metadata Demo")
    print("="*60)
    
    # Create metadata
    print("\n1. Creating model metadata...")
    metadata = ModelMetadata(
        version='v1.0.0',
        model_type='LSTM',
        training_date=datetime.now(),
        hyperparameters={'units_layer1': 128, 'units_layer2': 64, 'dropout': 0.2},
        feature_list=['Close', 'Volume', 'DXY', 'Oil', 'Treasury_10Y'],
        scaling_params={'min': 0, 'max': 1, 'method': 'minmax'},
        performance_metrics={'mae': 15.3, 'rmse': 22.1, 'r2': 0.92},
        training_data_range=(datetime(2020, 1, 1), datetime(2023, 12, 31)),
        sequence_length=60
    )
    
    print(f"   ✓ Version: {metadata.version}")
    print(f"   ✓ Model type: {metadata.model_type}")
    print(f"   ✓ Sequence length: {metadata.sequence_length}")
    print(f"   ✓ Features: {len(metadata.feature_list)}")
    
    # Convert to dictionary
    print("\n2. Converting metadata to dictionary...")
    metadata_dict = metadata.to_dict()
    
    print(f"   ✓ Dictionary keys: {list(metadata_dict.keys())}")
    print(f"   ✓ Performance metrics: {metadata_dict['performance_metrics']}")


def main():
    """Run all demos."""
    print("\n")
    print("="*60)
    print("ModelTrainingPipeline Demo")
    print("="*60)
    print("\nThis demo showcases the ModelTrainingPipeline class capabilities")
    print("for building various model architectures.")
    
    try:
        # Run demos
        demo_lstm_model()
        demo_gru_model()
        demo_xgboost_model()
        demo_random_forest_model()
        demo_model_metadata()
        
        # Summary
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\n✓ All model builders working correctly")
        print("✓ LSTM, GRU, XGBoost, and Random Forest models created")
        print("✓ Models can make predictions successfully")
        print("✓ Metadata system functional")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
