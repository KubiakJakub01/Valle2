from dataclasses import dataclass, field


@dataclass
class ValleHparams:
    # Input features
    vocab_size: int = field(default=256, metadata={'help': 'Vocab size'})
    num_audio_tokens: int = field(default=1024, metadata={'help': 'Number of audio tokens'})

    # Model
    d_model: int = field(default=256, metadata={'help': 'Model dimension'})
    n_head: int = field(default=4, metadata={'help': 'Number of heads'})
    dim_feedforward: int = field(default=1024, metadata={'help': 'Feedforward dimension'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate'})
    activation: str = field(default='relu', metadata={'help': 'Activation function'})
    num_layers: int = field(default=6, metadata={'help': 'Number of layers'})
