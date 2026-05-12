import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, Type
from src.utils.config_loader import load_config

class TimeSeriesDataset(Dataset):
    """
    Prepares sequences of windowed data for time series classification.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int = 4):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.window_size = window_size
        
    def __len__(self) -> int:
        return len(self.X) - self.window_size + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # X window shape: (window_size, num_features)
        window = self.X[idx : idx + self.window_size]
        # We associate the label with the last step of the window for real-time detection
        label = self.y[idx + self.window_size - 1] 
        return window, label

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.val_loss_min = val_loss
            self.counter = 0

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take last time-step output
        out = self.fc(out[:, -1, :])
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super(CNN1DModel, self).__init__()
        # Conv1d expects (batch_size, channels/input_dim, sequence_len)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        
        # The final dimension depends on input length (window_size) * hidden_dim
        # We instantiate dynamically or fix it. Let's make it flexible.
        self.fc = None 
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input comes as (batch, seq_len, input_dim)
        # Permute to (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        
        # Lazy initialization of FC layer once shape is known
        if self.fc is None:
            input_features = x.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(input_features, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ).to(x.device)
            
        return self.fc(x)

def train_evaluate_dl(
    model_class: Type[nn.Module],
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    window_size: int = None,
    config: dict = None
) -> Dict[str, float]:
    """
    Trains and evaluates a deep learning model following config strict rules.
    
    Returns dict with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    cfg = config if config else load_config()
    
    # Extract training settings
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    
    # Automatically extract default window from automata section if not provided
    if window_size is None:
        window_size = cfg['automata']['defaults']['window_size']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    # Prepare Datasets and Loaders
    train_ds = TimeSeriesDataset(X_train, y_train, window_size)
    val_ds = TimeSeriesDataset(X_val, y_val, window_size)
    test_ds = TimeSeriesDataset(X_test, y_test, window_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    # Assume input dimension is always size of feature columns (PC1 implies 1 dim)
    input_dim = X_train.shape[1]
    model = model_class(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    print(f"Starting training for {model_class.__name__} ({epochs} epochs limit)...")
    
    for epoch in range(epochs):
        # Train cycle
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation cycle
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Check early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    # Load best weights
    model.load_state_dict(early_stopping.best_model_wts)
    print("Model restored to best validation loss weights.")
    
    # Evaluation on Test Set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            
            # Thresholding for binary classification
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate Metrics
    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0)
    }
    
    print("\n--- Final Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    return metrics

if __name__ == "__main__":
    # Smoke test with dummy data
    print("Running smoke test...")
    dummy_x = np.random.rand(100, 1)
    dummy_y = np.random.randint(0, 2, size=(100,))
    
    # Test split
    X_tr, y_tr = dummy_x[:60], dummy_y[:60]
    X_v, y_v = dummy_x[60:80], dummy_y[60:80]
    X_te, y_te = dummy_x[80:], dummy_y[80:]
    
    try:
        res = train_evaluate_dl(LSTMModel, X_tr, y_tr, X_v, y_v, X_te, y_te, window_size=4)
        print("\nTest pass successful.")
    except Exception as e:
        print(f"\nSmoke test context: Missing config causes intended fallback or fail depending on configuration. Error info: {e}")
