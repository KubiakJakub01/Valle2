import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ConfigValle:
    # Data
    dataset: str = field(default='keithito/lj_speech', metadata={'help': 'Hugging Face dataset'})
    num_workers: int = field(default=4, metadata={'help': 'Number of workers'})

    # Input features
    vocab_size: int = field(default=256, metadata={'help': 'Vocab size'})
    num_audio_tokens: int = field(default=1024, metadata={'help': 'Number of audio tokens'})
    num_quantizers: int = field(
        default=8, metadata={'help': 'Number of quantizers layers from the audio codec'}
    )
    sampling_rate: int = field(default=16000, metadata={'help': 'Sampling rate'})
    polling_factor: int = field(default=320, metadata={'help': 'Polling factor'})

    # Model
    d_model: int = field(default=256, metadata={'help': 'Model dimension'})
    n_heads: int = field(default=4, metadata={'help': 'Number of heads'})
    dim_feedforward: int = field(default=1024, metadata={'help': 'Feedforward dimension'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate'})
    activation: Literal['relu', 'gelu'] = field(
        default='relu', metadata={'help': 'Activation function'}
    )
    num_layers: int = field(default=8, metadata={'help': 'Number of layers'})
    norm: Literal['AdaptiveLayerNorm', 'LayerNorm'] = field(
        default='AdaptiveLayerNorm', metadata={'help': 'Normalization layer'}
    )

    # Optimizer
    lr: float = field(default=1e-4, metadata={'help': 'Learning rate'})
    lr_warmup: int = field(default=1000, metadata={'help': 'Learning rate warmup steps'})
    betas: tuple = field(default=(0.9, 0.98), metadata={'help': 'Betas for Adam optimizer'})
    weight_decay: float = field(default=0.1, metadata={'help': 'Weight decay'})
    use_fused_adam: bool = field(default=True, metadata={'help': 'Use fused Adam optimizer'})
    gradient_clip_val: float = field(default=1.0, metadata={'help': 'Gradient clipping value'})
    grad_accum: int = field(default=1, metadata={'help': 'Gradient accumulation steps'})

    # Generation
    max_audio_len: int = field(default=1024, metadata={'help': 'Max length for generation'})
    num_beams: int = field(default=4, metadata={'help': 'Number of beams for generation'})
    use_kv_cache: bool = field(
        default=True, metadata={'help': 'Use key-value cache for generation'}
    )
    top_k: int = field(default=50, metadata={'help': 'Top-k for sampling'})
    tok_p: float = field(default=1.0, metadata={'help': 'Token probability'})
    temperature: float = field(default=1.0, metadata={'help': 'Temperature'})
    length_penalty: float = field(default=1.0, metadata={'help': 'Length penalty'})

    # Training
    seed: int = field(default=42, metadata={'help': 'Seed for reproducibility'})
    batch_size: int = field(default=4, metadata={'help': 'Batch size'})
    valid_batch_size: int = field(default=1, metadata={'help': 'Validation batch size'})
    max_steps: int = field(default=1000, metadata={'help': 'Max steps'})
    log_every_n_steps: int = field(default=100, metadata={'help': 'Log every n steps'})
    ckpt_path: Path = field(
        default=Path('models/checkpoints'), metadata={'help': 'Checkpoint path'}
    )
    log_path: Path = field(default=Path('models/logs'), metadata={'help': 'Log path'})

    def __post_init__(self):
        if self.dataset is None:
            raise ValueError('Dataset must be provided')
        if self.norm not in ['AdaptiveLayerNorm', 'LayerNorm']:
            raise ValueError('Normalization layer must be AdaptiveLayerNorm or LayerNorm')
        if self.activation not in ['relu', 'gelu']:
            raise ValueError('Activation function must be relu or gelu')

        self.ckpt_path = Path(self.ckpt_path)
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(self.log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

    @property
    def quantization_factor(self):
        return self.sampling_rate // self.polling_factor

    @property
    def bos_token(self):
        return self.num_audio_tokens + 1

    @property
    def eos_token(self):
        return self.num_audio_tokens

    @classmethod
    def from_dict(cls, hparams_dict):
        return cls(**hparams_dict)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, encoding='utf-8') as f:
            hparams_dict = json.load(f)
        return cls.from_dict(hparams_dict)
