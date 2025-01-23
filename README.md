# Toronto Weather Forecasting System

A deep learning-based weather prediction system that leverages LSTM neural networks to forecast multiple weather parameters simultaneously.

## Overview

This project implements an advanced weather forecasting system that:
- Takes historical weather data as input
- Uses deep learning to identify patterns
- Predicts multiple weather parameters for future time steps
- Provides detailed visualization of predictions

## Technical Details

### Data Processing
The system processes weather data from `toronto_weather_data.csv`, which includes:
- Multiple input features (temperature, humidity, etc.)
- Historical weather measurements
- Time-based information (month, day, hour)

### Model Architecture
- Type: Multi-layer LSTM (Long Short-Term Memory)
- Features:
  * Multiple LSTM layers with dropout
  * Dense output layers
  * Multi-step prediction capability
  * Configurable prediction horizon

### Key Parameters
```python
sequence_length = 10    # Past time steps used for prediction
forecast_horizon = 6    # Future time steps to predict
hidden_size = 50       # LSTM hidden layer size
dropout_rate = 0.1     # Dropout for regularization
```

### Performance Metrics
The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) coefficient

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Install required packages
   pip install numpy pandas torch scikit-learn matplotlib
   ```

2. **Data Preparation**
   - Place `toronto_weather_data.csv` in the project root
   - Ensure data format matches expected schema

3. **Running the Model**
   ```bash
   python weather_forecaster.py
   ```

## Output

The system generates:
1. Training progress metrics
2. Prediction accuracy measurements
3. Visualization plots comparing:
   - Actual vs predicted values
   - Training and testing performance
   - Multi-step forecast accuracy

## Dependencies
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure
```
.
├── weather_forecaster.py    # Main model implementation
├── toronto_weather_data.csv # Dataset
└── README.md               # Documentation
```
