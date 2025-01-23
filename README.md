# Advanced Weather Forecasting System

A sophisticated deep learning system for multi-step weather prediction using bidirectional LSTM networks with enhanced architecture and training procedures.

## Technical Architecture

### Model Components
- **Bidirectional LSTM**:
  * Multi-layer architecture (2 layers)
  * Layer normalization
  * Residual connections
  * Dropout regularization
  * Advanced fully connected layers

### Key Features
- Multi-step prediction (6-hour forecast)
- 24-hour lookback window
- Separate feature/target normalization
- Early stopping mechanism
- Learning rate scheduling
- Gradient clipping
- Comprehensive metrics tracking

## Model Configuration

```python
CONFIG = {
    "sequence_length": 24,     # Hours of historical data used
    "forecast_horizon": 6,     # Hours to predict
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 10,           # Early stopping patience
    "validation_split": 0.2
}
```

## Data Processing Pipeline

### Preprocessing
1. **Data Loading**:
   - Loads weather data from CSV
   - Separates features and targets
   - Applies independent normalization

2. **Sequence Creation**:
   - Sliding window approach
   - Configurable sequence length
   - Efficient numpy-based processing

3. **Data Splitting**:
   - 60% Training
   - 20% Validation
   - 20% Testing

### Model Architecture
```
Input → Bidirectional LSTM → Layer Normalization → 
Dense (256) → ReLU → Dropout → Dense (output) → 
Reshape (forecast steps)
```

## Training Process

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-3 with scheduling
- **Loss Function**: MSE
- **Regularization**: 
  * Dropout (20%)
  * Weight decay (1e-4)
  * Gradient clipping

### Monitoring
- Training/Validation loss tracking
- Early stopping with patience
- Model checkpointing
- Learning rate adjustment

## Performance Metrics
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) coefficient
- Per-variable prediction plots

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Setup and Usage

1. **Environment Setup**:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib
   ```

2. **Data Preparation**:
   - Place `toronto_weather_data.csv` in project root
   - Ensure data format matches expected schema

3. **Running the Model**:
   ```bash
   python weather_forecaster.py
   ```

## Output Visualization
- Time series plots for each target variable
- Actual vs. predicted comparisons
- Training/validation loss curves
- Comprehensive metric reporting

## Project Structure
```
.
├── weather_forecaster.py    # Main implementation
├── toronto_weather_data.csv # Dataset
└── README.md               # Documentation
```

## Model Artifacts
- `best_model.pth`: Best model checkpoint
- Training history
- Performance metrics
- Prediction visualizations
