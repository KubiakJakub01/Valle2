import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ValleHparams:
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
    n_head: int = field(default=4, metadata={'help': 'Number of heads'})
    dim_feedforward: int = field(default=1024, metadata={'help': 'Feedforward dimension'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate'})
    activation: Literal['relu', 'gelu'] = field(
        default='relu', metadata={'help': 'Activation function'}
    )
    num_layers: int = field(default=6, metadata={'help': 'Number of layers'})
    norm: Literal['AdaptiveLayerNorm', 'LayerNorm'] = field(
        default='AdaptiveLayerNorm', metadata={'help': 'Normalization layer'}
    )

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
