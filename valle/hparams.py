import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ValleHparams:
    # Dataset
    dataset: str = field(default='theodorr/ljspeech', metadata={'help': 'Hugging Face dataset'})

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
    num_layers: int = field(default=6, metadata={'help': 'Number of layers'})
    norm: Literal['AdaptiveLayerNorm', 'LayerNorm'] = field(
        default='AdaptiveLayerNorm', metadata={'help': 'Normalization layer'}
    )

    # Optimizer
    lr: float = field(default=1e-4, metadata={'help': 'Learning rate'})
    betas: tuple = field(default=(0.9, 0.98), metadata={'help': 'Betas for Adam optimizer'})
    weight_decay: float = field(default=0.1, metadata={'help': 'Weight decay'})
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
    max_steps: int = field(default=1000, metadata={'help': 'Max steps'})
    log_every_n_steps: int = field(default=100, metadata={'help': 'Log every n steps'})
    ckpt_path: str = field(default='checkpoints', metadata={'help': 'Checkpoint path'})

    @property
    def quantization_factor(self):
        return self.sampling_rate // self.polling_factor

    @classmethod
    def from_dict(cls, hparams_dict):
        return cls(**hparams_dict)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, encoding='utf-8') as f:
            hparams_dict = json.load(f)
        return cls.from_dict(hparams_dict)
