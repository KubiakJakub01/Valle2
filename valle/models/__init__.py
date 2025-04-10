from pathlib import Path

import lightning as L
import torch

from ..config import ConfigValle
from ..utils import log_error, log_info
from .encodec_pip import EncodecPip
from .valle_ar import ValleAR
from .valle_nar import ValleNAR

MODEL_DICT = {
    'encodec_pip': EncodecPip,
    'valle_ar': ValleAR,
    'valle_nar': ValleNAR,
}


def get_model_class(model_name: str):
    return MODEL_DICT[model_name]


def load_model_for_inference(
    checkpoint_path: Path,
    model_class: type[L.LightningModule],
    config: ConfigValle,
    device: torch.device,
) -> L.LightningModule:
    """Load a Lightning model from a checkpoint."""
    log_info(f'Loading {model_class.__name__} from checkpoint: {checkpoint_path}')
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path, map_location=device, config=config
        )
        model.eval()
        model.to(device)
        log_info(f'{model_class.__name__} loaded successfully.')
        return model
    except FileNotFoundError:
        log_error(f'Checkpoint file not found: {checkpoint_path}')
        raise
    except Exception as e:
        log_error(f'Error loading model from {checkpoint_path}: {e}')
        raise


__all__ = ['encodec_pip', 'valle_ar', 'valle_nar', 'get_model_class', 'load_model_for_inference']
