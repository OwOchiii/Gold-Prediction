# Gold Price Prediction System

A comprehensive machine learning system for forecasting gold (XAU) prices using historical OHLCV data and economic indicators.

## Project Structure

```
Gold Prediction/
├── data/               # Raw and processed data files
├── models/             # Trained model files and metadata
├── reports/            # Evaluation reports and visualizations
├── src/                # Source code modules
│   ├── __init__.py
│   └── logger.py       # Logging configuration
├── tests/              # Unit and integration tests
│   └── __init__.py
├── logs/               # Application log files
├── config.py           # System configuration parameters
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Features

- **Data Ingestion**: Load OHLCV data from CSV and economic indicators via yfinance API
- **Data Preprocessing**: Handle missing values, normalize features, remove outliers
- **Feature Engineering**: Create lag features, rolling statistics, technical indicators
- **Multiple Models**: Support for LSTM, GRU, XGBoost, and Random Forest models
- **Prediction Service**: Generate single-step and multi-step forecasts with confidence intervals
- **Model Evaluation**: Comprehensive metrics (MAE, RMSE, MAPE, R², directional accuracy)
- **Visualization**: Time series plots, feature importance, and prediction reports

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All system parameters are defined in `config.py`:

- **Data paths**: DATA_DIR, MODEL_DIR, REPORTS_DIR
- **Model parameters**: SEQUENCE_LENGTH, TRAIN_RATIO, EPOCHS, LEARNING_RATE
- **Feature engineering**: LAG_PERIODS, ROLLING_WINDOWS, RSI_PERIOD
- **Economic indicators**: DXY, Oil, Treasury yields
- **Quality thresholds**: MAX_MISSING_PCT, DRIFT_THRESHOLD

## Usage

### Training a Model

```python
from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainingPipeline

# Load and preprocess data
data_manager = DataIngestionManager()
df = data_manager.load_csv('data/gold_prices.csv')

# Train model
pipeline = ModelTrainingPipeline()
model = pipeline.train_model(df)
```

### Generating Predictions

```python
from src.prediction import PredictionService

# Load trained model and generate predictions
predictor = PredictionService()
predictions = predictor.predict_multi_step(input_data, horizon=30)
```

## Technology Stack

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning models (LSTM, GRU)
- **scikit-learn**: Traditional ML models and metrics
- **XGBoost**: Gradient boosting model
- **pandas/numpy**: Data processing
- **yfinance**: Economic indicator data download
- **matplotlib/seaborn**: Visualization
- **pandas-ta**: Technical indicators

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test module:
```bash
pytest tests/test_data_ingestion.py -v
```

## Logging

Logs are automatically created in the `logs/` directory. Configure logging in `src/logger.py` or through Config class:

```python
from src.logger import get_logger

logger = get_logger(__name__)
logger.info('Processing started')
```

## Model Versioning

Trained models are stored in `models/` with version identifiers:
```
models/
├── model_v1.0.0/
│   ├── model.h5          # Keras model
│   ├── metadata.json     # Training metadata
│   ├── scaler.pkl        # Normalization parameters
│   └── feature_list.json # Feature names
└── registry.json         # Model registry
```

## Development Roadmap

- [x] Project structure and configuration setup
- [ ] Data ingestion and validation module
- [ ] Data preprocessing and cleaning module
- [ ] Feature engineering module
- [ ] Model training pipeline
- [ ] Prediction service
- [ ] Model evaluation and visualization
- [ ] End-to-end pipeline orchestration

## License

Copyright © 2024 Gold Prediction System Team

## Contributors

Gold Prediction System Development Team
