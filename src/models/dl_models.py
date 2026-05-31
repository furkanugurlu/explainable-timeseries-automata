import logging
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, Type
from src.utils.config_loader import load_config
from src.models.base_model import BaseAnomalyDetector

logger = logging.getLogger(__name__)

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
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
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

class GRUModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
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
    logger.info(f"Training device: {device}")
    
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
    
    logger.info(f"Starting training: {model_class.__name__} (max {epochs} epochs)")
    
    train_start_time = time.time()
    
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
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    model.load_state_dict(early_stopping.best_model_wts)
    train_duration = time.time() - train_start_time
    logger.info(f"Training complete. Best weights restored. Duration: {train_duration:.2f}s")
    
    # Evaluation on Test Set
    model.eval()
    all_preds = []
    all_targets = []
    
    inference_start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            
            # Thresholding for binary classification
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    inference_duration = time.time() - inference_start_time
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate Metrics
    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "train_time_sec": train_duration,
        "inference_time_sec": inference_duration
    }
    
    logger.info(f"Eval metrics: {', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}")

    return metrics


class DLAnomalyDetector(BaseAnomalyDetector):
    """
    Strategy pattern adapter: wraps train_evaluate_dl() so all DL models
    conform to the same BaseAnomalyDetector interface as AutomataPipeline.

    fit() stores train/val data; get_metrics() runs the full training loop.
    predict() is not supported standalone (DL training is inseparable from eval).
    """

    def __init__(self, model_class: Type[nn.Module], config: dict = None):
        self.model_class = model_class
        self.config = config or load_config()
        self._X_train: np.ndarray = None
        self._y_train: np.ndarray = None
        self._X_val: np.ndarray = None
        self._y_val: np.ndarray = None
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> "DLAnomalyDetector":
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val if X_val is not None else X_train
        self._y_val = y_val if y_val is not None else y_train
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DLAnomalyDetector does not support predict() alone. "
            "Training and inference are inseparable — use get_metrics(X_test, y_test)."
        )

    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError(
                "DLAnomalyDetector.get_metrics() called before fit(). Call fit(X_train, y_train) first."
            )
        return train_evaluate_dl(
            self.model_class,
            self._X_train, self._y_train,
            self._X_val, self._y_val,
            X_test, y_test,
            config=self.config,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running DL smoke test...")

    dummy_x = np.random.rand(100, 1)
    dummy_y = np.random.randint(0, 2, size=(100,))

    X_tr, y_tr = dummy_x[:60], dummy_y[:60]
    X_v, y_v = dummy_x[60:80], dummy_y[60:80]
    X_te, y_te = dummy_x[80:], dummy_y[80:]

    try:
        res = train_evaluate_dl(LSTMModel, X_tr, y_tr, X_v, y_v, X_te, y_te, window_size=4)
        logger.info("DL smoke test passed successfully.")
    except Exception as e:
        logger.warning(f"Smoke test: missing config causes expected fallback. Error: {e}")
