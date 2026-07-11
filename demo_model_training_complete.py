"""
Demonstration script for Model Training Pipeline with all features.

This script demonstrates:
1. Basic model training with early stopping and checkpointing
2. Training for tree-based models (XGBoost, Random Forest)
3. Training metrics logging
4. Hyperparameter tuning

Requirements: 5.1, 5.2, 5.4, 5.6
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model_training import ModelTrainingPipeline, ModelMetadata
from config import Config
from datetime import datetime


def demo_lstm_training_with_checkpointing():
    """Demonstrate LSTM training with early stopping and checkpointing."""
    print("=" * 80)
    print("DEMO 1: LSTM Training with Early Stopping and Checkpointing")
    print("=" * 80)
    
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic sequence data
    np.random.seed(42)
    n_samples = 500
    sequence_length = 30
    n_features = 8
    
    print(f"\n1. Generating synthetic data:")
    print(f"   - Training samples: 350")
    print(f"   - Validation samples: 150")
    print(f"   - Sequence length: {sequence_length}")
    print(f"   - Features: {n_features}")
    
    # Create training and validation data
    X_train = np.random.randn(350, sequence_length, n_features)
    y_train = np.random.randn(350)
    X_val = np.random.randn(150, sequence_length, n_features)
    y_val = np.random.randn(150)
    
    print(f"\n2. Building LSTM model...")
    model = pipeline.build_lstm_model(input_shape=(sequence_length, n_features))
    print(f"   Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Create checkpoint directory
    checkpoint_dir = Config.MODEL_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "lstm_best.keras"
    
    print(f"\n3. Training model with early stopping (patience=10)...")
    print(f"   Checkpoint path: {checkpoint_path}")
    print(f"   Max epochs: {Config.EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    
    # Train with checkpointing
    result = pipeline.train_model(
        model, X_train, y_train, X_val, y_val,
        model_type='deep_learning',
        checkpoint_path=str(checkpoint_path)
    )
    
    print(f"\n4. Training Results:")
    print(f"   - Training time: {result.training_time:.2f} seconds")
    print(f"   - Epochs trained: {len(result.history['loss'])}")
    print(f"   - Final training loss: {result.final_loss:.6f}")
    print(f"   - Final validation loss: {result.validation_loss:.6f}")
    print(f"   - Convergence status: {result.convergence_status}")
    print(f"   - Checkpoint saved: {checkpoint_path.exists()}")
    
    # Log training metrics
    print(f"\n5. Logging training metrics...")
    pipeline.log_training_metrics(result, 'demo_lstm_v1.0', save_to_file=True)
    
    history_file = Config.MODEL_DIR / 'training_history_demo_lstm_v1.0.json'
    print(f"   Training history saved to: {history_file}")
    print(f"   History file exists: {history_file.exists()}")


def demo_xgboost_training():
    """Demonstrate XGBoost training with eval_set."""
    print("\n" + "=" * 80)
    print("DEMO 2: XGBoost Training with Early Stopping via eval_set")
    print("=" * 80)
    
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic tabular data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    print(f"\n1. Generating synthetic tabular data:")
    print(f"   - Training samples: 700")
    print(f"   - Validation samples: 300")
    print(f"   - Features: {n_features}")
    
    X_train = np.random.randn(700, n_features)
    y_train = np.random.randn(700)
    X_val = np.random.randn(300, n_features)
    y_val = np.random.randn(300)
    
    print(f"\n2. Building XGBoost model...")
    hyperparams = {
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.1
    }
    model = pipeline.build_xgboost_model(hyperparams)
    print(f"   Model configuration:")
    print(f"   - Max depth: {hyperparams['max_depth']}")
    print(f"   - N estimators: {hyperparams['n_estimators']}")
    print(f"   - Learning rate: {hyperparams['learning_rate']}")
    
    print(f"\n3. Training XGBoost model...")
    result = pipeline.train_model(
        model, X_train, y_train, X_val, y_val,
        model_type='tree_based'
    )
    
    print(f"\n4. Training Results:")
    print(f"   - Training time: {result.training_time:.2f} seconds")
    print(f"   - Final training loss (MSE): {result.final_loss:.6f}")
    print(f"   - Final validation loss (MSE): {result.validation_loss:.6f}")
    print(f"   - Convergence status: {result.convergence_status}")


def demo_random_forest_training():
    """Demonstrate Random Forest training."""
    print("\n" + "=" * 80)
    print("DEMO 3: Random Forest Training")
    print("=" * 80)
    
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic data
    np.random.seed(42)
    n_features = 12
    
    print(f"\n1. Generating synthetic data:")
    print(f"   - Training samples: 500")
    print(f"   - Validation samples: 200")
    print(f"   - Features: {n_features}")
    
    X_train = np.random.randn(500, n_features)
    y_train = np.random.randn(500)
    X_val = np.random.randn(200, n_features)
    y_val = np.random.randn(200)
    
    print(f"\n2. Building Random Forest model...")
    hyperparams = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5
    }
    model = pipeline.build_random_forest_model(hyperparams)
    print(f"   Model configuration:")
    print(f"   - N estimators: {hyperparams['n_estimators']}")
    print(f"   - Max depth: {hyperparams['max_depth']}")
    print(f"   - Min samples split: {hyperparams['min_samples_split']}")
    
    print(f"\n3. Training Random Forest model...")
    result = pipeline.train_model(
        model, X_train, y_train, X_val, y_val,
        model_type='tree_based'
    )
    
    print(f"\n4. Training Results:")
    print(f"   - Training time: {result.training_time:.2f} seconds")
    print(f"   - Final training loss (MSE): {result.final_loss:.6f}")
    print(f"   - Final validation loss (MSE): {result.validation_loss:.6f}")
    print(f"   - Convergence status: {result.convergence_status}")


def demo_hyperparameter_tuning_xgboost():
    """Demonstrate hyperparameter tuning for XGBoost."""
    print("\n" + "=" * 80)
    print("DEMO 4: Hyperparameter Tuning - XGBoost with GridSearchCV")
    print("=" * 80)
    
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic data
    np.random.seed(42)
    n_features = 10
    
    print(f"\n1. Generating synthetic data:")
    print(f"   - Training samples: 400")
    print(f"   - Validation samples: 100")
    print(f"   - Features: {n_features}")
    
    X_train = np.random.randn(400, n_features)
    y_train = np.random.randn(400)
    X_val = np.random.randn(100, n_features)
    y_val = np.random.randn(100)
    
    # Define small search space for demo
    search_space = {
        'max_depth': [3, 5],
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1]
    }
    
    print(f"\n2. Hyperparameter search space:")
    for param, values in search_space.items():
        print(f"   - {param}: {values}")
    
    total_configs = 1
    for values in search_space.values():
        total_configs *= len(values)
    print(f"\n   Total configurations to test: {total_configs}")
    
    print(f"\n3. Running GridSearchCV...")
    best_config = pipeline.hyperparameter_tuning(
        'XGBoost', X_train, y_train, X_val, y_val,
        search_space=search_space
    )
    
    print(f"\n4. Tuning Results:")
    print(f"   Best hyperparameters:")
    for param, value in best_config['best_params'].items():
        print(f"   - {param}: {value}")
    print(f"   Best validation loss: {best_config['best_val_loss']:.6f}")


def demo_hyperparameter_tuning_lstm():
    """Demonstrate hyperparameter tuning for LSTM."""
    print("\n" + "=" * 80)
    print("DEMO 5: Hyperparameter Tuning - LSTM with Manual Loop")
    print("=" * 80)
    
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic sequence data
    np.random.seed(42)
    sequence_length = 20
    n_features = 5
    
    print(f"\n1. Generating synthetic sequence data:")
    print(f"   - Training samples: 200")
    print(f"   - Validation samples: 100")
    print(f"   - Sequence length: {sequence_length}")
    print(f"   - Features: {n_features}")
    
    X_train = np.random.randn(200, sequence_length, n_features)
    y_train = np.random.randn(200)
    X_val = np.random.randn(100, sequence_length, n_features)
    y_val = np.random.randn(100)
    
    # Define small search space for demo (to keep it fast)
    search_space = {
        'units_layer1': [64, 128],
        'units_layer2': [32, 64],
        'dropout': [0.2],
        'learning_rate': [0.001]
    }
    
    print(f"\n2. Hyperparameter search space:")
    for param, values in search_space.items():
        print(f"   - {param}: {values}")
    
    total_configs = 1
    for values in search_space.values():
        total_configs *= len(values)
    print(f"\n   Total configurations to test: {total_configs}")
    
    print(f"\n3. Running manual configuration loop...")
    print(f"   Note: Each configuration trains for up to {Config.EPOCHS} epochs")
    print(f"   with early stopping patience={Config.EARLY_STOPPING_PATIENCE}")
    
    best_config = pipeline.hyperparameter_tuning(
        'LSTM', X_train, y_train, X_val, y_val,
        search_space=search_space
    )
    
    print(f"\n4. Tuning Results:")
    print(f"   Best hyperparameters:")
    for param, value in best_config['best_params'].items():
        print(f"   - {param}: {value}")
    print(f"   Best validation loss: {best_config['best_val_loss']:.6f}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 80)
    print("# Model Training Pipeline - Complete Functionality Demo")
    print("#" * 80)
    
    try:
        # Demo 1: LSTM with checkpointing
        demo_lstm_training_with_checkpointing()
        
        # Demo 2: XGBoost training
        demo_xgboost_training()
        
        # Demo 3: Random Forest training
        demo_random_forest_training()
        
        # Demo 4: Hyperparameter tuning for XGBoost
        demo_hyperparameter_tuning_xgboost()
        
        # Demo 5: Hyperparameter tuning for LSTM (small search space)
        demo_hyperparameter_tuning_lstm()
        
        print("\n" + "=" * 80)
        print("All demonstrations completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
