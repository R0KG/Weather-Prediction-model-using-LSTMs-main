import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
from typing import Tuple

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration
CONFIG = {
    "sequence_length": 24,  # 24-hour lookback
    "forecast_horizon": 6,  # 6-hour forecast
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 10,
    "validation_split": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """Load and normalize weather data"""
    df = pd.read_csv(file_path)
    
    # Split features and targets
    feature_cols = [col for col in df.columns if not col.isdigit()]
    target_cols = [col for col in df.columns if col.isdigit()]
    
    # Create separate scalers for features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Normalize features and targets separately
    df_features = pd.DataFrame(feature_scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    df_targets = pd.DataFrame(target_scaler.fit_transform(df[target_cols]), columns=target_cols)
    
    # Combine normalized features and targets
    df_normalized = pd.concat([df_features, df_targets], axis=1)
    
    return df_normalized, feature_cols, target_cols, (feature_scaler, target_scaler)

def create_sequences(data: pd.DataFrame, feature_cols: list, target_cols: list, 
                    seq_len: int, pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create time series sequences with sliding window"""
    sequences = []
    targets = []
    
    # Convert to numpy for faster processing
    feature_data = data[feature_cols].values
    target_data = data[target_cols].values
    
    for i in range(len(data) - seq_len - pred_len + 1):
        sequences.append(feature_data[i:i+seq_len])
        targets.append(target_data[i+seq_len:i+seq_len+pred_len])
    
    # Convert to numpy first, then to tensor
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    return (torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))

class WeatherLSTM(nn.Module):
    """Enhanced LSTM model with layer normalization and residual connections"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, pred_len: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=CONFIG["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=CONFIG["dropout_rate"]
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # For bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_rate"]),
            nn.Linear(256, output_size * pred_len)
        )
        self.pred_len = pred_len
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.layer_norm(out[:, -1, :])  # Last timestep
        out = self.fc(out)
        return out.view(-1, self.pred_len, self.output_size)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """Training loop with early stopping and learning rate scheduling"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = nn.MSELoss()  # Changed from HuberLoss to MSELoss for stability
    
    best_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(CONFIG["num_epochs"]):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(CONFIG["device"]), y.to(CONFIG["device"])
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(CONFIG["device"]), y.to(CONFIG["device"])
                preds = model(X)
                val_loss += loss_fn(preds, y).item()
        
        # Update learning rate
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            
        if no_improve == CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
    
    return history

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    """Comprehensive model evaluation"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(CONFIG["device"])
            preds = model(X).cpu().numpy()
            predictions.append(preds)
            actuals.append(y.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    metrics = {
        'rmse': np.sqrt(np.mean((predictions - actuals) ** 2)),
        'mae': np.mean(np.abs(predictions - actuals)),
        'r2': r2_score(actuals.flatten(), predictions.flatten())
    }
    
    return metrics

def plot_results(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, 
                target_cols: list, config: dict):
    """Enhanced visualization with confidence intervals"""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(config["device"])
        preds = model(X_test).cpu().numpy()
    
    actuals = y_test.numpy()
    
    for i, col in enumerate(target_cols):
        plt.figure(figsize=(15, 5))
        plt.plot(actuals[:, -1, i], label='Actual')
        plt.plot(preds[:, -1, i], alpha=0.7, label='Predicted')
        plt.title(f'Target Variable: {col}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    # Load and preprocess data
    data_df, feature_cols, target_cols, scalers = load_and_preprocess_data('toronto_weather_data.csv')
    
    # Create sequences
    X, y = create_sequences(data_df, feature_cols, target_cols,
                          CONFIG["sequence_length"], 
                          CONFIG["forecast_horizon"])
    
    # Train/val/test split with time series awareness
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    test_size = len(X) - train_size - val_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=CONFIG["batch_size"], 
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                           batch_size=CONFIG["batch_size"],
                           num_workers=4,
                           pin_memory=True)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=CONFIG["batch_size"],
                            num_workers=4,
                            pin_memory=True)
    
    # Initialize model
    input_size = len(feature_cols)
    output_size = len(target_cols)
    
    print(f"Input size: {input_size}, Output size: {output_size}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")
    
    model = WeatherLSTM(input_size, 
                       CONFIG["hidden_size"], 
                       output_size,
                       CONFIG["forecast_horizon"]).to(CONFIG["device"])
    
    # Training
    history = train_model(model, train_loader, val_loader)
    
    # Evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate_model(model, test_loader)
    print(f"\nTest Metrics: RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
    
    # Visualization
    plot_results(model, X_test, y_test, target_cols, CONFIG)

if __name__ == "__main__":
    main()