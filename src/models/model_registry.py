from src.models.dl_models import LSTMModel, GRUModel, CNN1DModel

# Single source of truth for all DL model backbone classes.
# main.py wraps each entry in DLAnomalyDetector — no direct instantiation here.
MODEL_REGISTRY: dict = {
    "LSTM":  LSTMModel,
    "GRU":   GRUModel,
    "CNN1D": CNN1DModel,
}


def get_model_class(name: str):
    """Returns the nn.Module backbone class for the given name (case-sensitive)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
