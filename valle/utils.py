import logging
from collections.abc import Callable

import coloredlogs
import torchaudio
from torch import Tensor

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    coloredlogs.ColoredFormatter(
        fmt='%(asctime)s :: %(levelname)s :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log_debug(*args, **kwargs):
    """Log an debug message."""
    logger.debug(*args, **kwargs)


def log_info(*args, **kwargs):
    """Log an info message."""
    logger.info(*args, **kwargs)


def log_warning(*args, **kwargs):
    """Log a warning message."""
    logger.warning(*args, **kwargs)


def log_error(*args, **kwargs):
    """Log an error message."""
    logger.error(*args, **kwargs)


def tree_map(fn: Callable, x):
    if isinstance(x, list):
        x = [tree_map(fn, xi) for xi in x]
    elif isinstance(x, tuple):
        x = (tree_map(fn, xi) for xi in x)
    elif isinstance(x, dict):
        x = {k: tree_map(fn, v) for k, v in x.items()}
    elif isinstance(x, Tensor):
        x = fn(x)
    return x


def to_device(x, device):
    return tree_map(lambda t: t.to(device), x)


def normalize_audio(audio: Tensor, orginal_sr: int, target_sr: int = 16000) -> Tensor:
    """Normalize audio to target sample rate."""
    # Normalize to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    # Normalize to target sample rate
    if orginal_sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=orginal_sr, new_freq=target_sr)(audio)
    # Normalize to [-1, 1]
    audio = audio / audio.abs().max()
    return audio
