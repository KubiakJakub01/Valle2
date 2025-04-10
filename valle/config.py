import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal


@dataclass
class ConfigValle:
    # Data
    data_dir: Path = field(
        default=Path('data'),
        metadata={'help': 'Path to the directory containing prepared data'},
    )
    language: str = field(
        default='en-us', metadata={'help': 'Language (used for phonemizer consistency check)'}
    )
    num_workers: int = field(default=4, metadata={'help': 'Number of workers for DataLoader'})

    # Input features
    num_audio_tokens: int = field(
        default=1024, metadata={'help': 'Number of audio tokens (codebook size)'}
    )
    num_quantizers: int = field(
        default=8, metadata={'help': 'Number of quantizers layers from the audio codec'}
    )
    sampling_rate: int = field(default=24000, metadata={'help': 'Sampling rate'})
    polling_factor: int = field(default=320, metadata={'help': 'Polling factor'})

    # Model
    model_name: Literal['valle_ar', 'valle_nar'] = field(
        default='valle_ar', metadata={'help': 'Model name'}
    )
    d_model: int = field(default=256, metadata={'help': 'Model dimension'})
    n_heads: int = field(default=4, metadata={'help': 'Number of heads'})
    dim_feedforward: int = field(default=1024, metadata={'help': 'Feedforward dimension'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate'})
    activation: Literal['relu', 'gelu'] = field(
        default='relu', metadata={'help': 'Activation function'}
    )
    num_layers: int = field(default=8, metadata={'help': 'Number of layers'})

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
    base_checkpoint: int | None = field(
        default=None, metadata={'help': 'Base checkpoint to resume from'}
    )
    seed: int = field(default=42, metadata={'help': 'Seed for reproducibility'})
    batch_size: int = field(default=4, metadata={'help': 'Batch size'})
    valid_batch_size: int = field(default=1, metadata={'help': 'Validation batch size'})
    max_steps: int = field(default=10000, metadata={'help': 'Max steps'})
    steps_per_log: int = field(default=100, metadata={'help': 'Log every n steps'})
    steps_per_ckpt: int = field(default=1000, metadata={'help': 'Checkpoint every n steps'})
    ckpt_path: Path = field(
        default=Path('models/checkpoints'), metadata={'help': 'Checkpoint path'}
    )
    log_path: Path = field(default=Path('models/logs'), metadata={'help': 'Log path'})

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).absolute()
        self.ckpt_path = Path(self.ckpt_path).absolute()
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(self.log_path).absolute()
        self.log_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return json.dumps(self.__dict__, default=str, indent=4)

    @cached_property
    def phoneme_to_id(self):
        with open(self.vocab_path, encoding='utf-8') as f:
            vocab_data = json.load(f)
        return {phoneme: int(id_str) for id_str, phoneme in vocab_data.items()}

    @cached_property
    def vocab_size(self):
        return len(self.phoneme_to_id) + 1

    @property
    def checkpoint_path(self):
        if self.base_checkpoint is None:
            raise ValueError('Base checkpoint is not set')
        return self.ckpt_path / f'step={self.base_checkpoint}.ckpt'

    @property
    def metadata_path(self):
        metadata_path = self.data_dir / 'metadata.tsv'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Metadata file not found: {metadata_path}')
        return metadata_path

    @property
    def vocab_path(self):
        vocab_path = self.data_dir / 'vocab.json'
        if not vocab_path.is_file():
            raise FileNotFoundError(f'Vocabulary file not found: {vocab_path}')
        return vocab_path

    @property
    def codes_dir(self):
        codes_dir = self.data_dir / 'codes'
        if not codes_dir.is_dir():
            raise NotADirectoryError(f'Codes directory not found: {codes_dir}')
        return codes_dir

    @property
    def norm(self):
        if self.model_name == 'valle_ar':
            return 'LayerNorm'
        return 'AdaptiveLayerNorm'

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
        if 'data_dir' in hparams_dict and isinstance(hparams_dict['data_dir'], str):
            hparams_dict['data_dir'] = Path(hparams_dict['data_dir'])
        if 'ckpt_path' in hparams_dict and isinstance(hparams_dict['ckpt_path'], str):
            hparams_dict['ckpt_path'] = Path(hparams_dict['ckpt_path'])
        if 'log_path' in hparams_dict and isinstance(hparams_dict['log_path'], str):
            hparams_dict['log_path'] = Path(hparams_dict['log_path'])
        return cls(**hparams_dict)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, encoding='utf-8') as f:
            hparams_dict = json.load(f)
        return cls.from_dict(hparams_dict)

    def dump_to_json(self, json_file: Path):
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, default=str, indent=4)
