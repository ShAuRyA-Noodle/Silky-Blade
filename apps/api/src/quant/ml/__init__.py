"""End-to-end LightGBM trainer (features + triple-barrier labels + purged K-fold + MLflow)."""

from quant.ml.config import TrainConfig, load_config
from quant.ml.trainer import train

__all__ = ["TrainConfig", "load_config", "train"]
