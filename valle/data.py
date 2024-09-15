from typing import Literal

import torch
from datasets import load_dataset
from einops import rearrange
from g2p_en import G2p
from torch.utils.data import DataLoader, Dataset

from .collate import get_collate
from .config import ConfigValle
from .models import EncodecPip
from .utils import normalize_audio


class ValleDataset(Dataset):
    def __init__(self, dataset, config: ConfigValle):
        self.dataset = dataset
        self.config = config
        self.encodec_pip = EncodecPip()
        self.g2p = G2p()
        self.sym2idx = {sym: idx for idx, sym in enumerate(self.g2p.phonemes)}
        self.sym2idx[' '] = len(self.sym2idx)
        self.sym2idx[','] = len(self.sym2idx)
        self.sym2idx['.'] = len(self.sym2idx)

    def _tokenize(self, text: str) -> torch.Tensor:
        return torch.tensor([self.sym2idx[phoneme] for phoneme in self.g2p(text)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load item
        item = self.dataset[idx]
        audio = rearrange(torch.tensor(item['audio']['array'], dtype=torch.float32), 't -> 1 t')
        sr = item['audio']['sampling_rate']
        tokens = self._tokenize(item['text'])

        # Normalize audio
        audio = rearrange(normalize_audio(audio, sr, self.encodec_pip.sampling_rate), '1 t -> t')

        # Encode audio
        codes = self.encodec_pip.encode(audio)

        return {'codes': codes, 'tokens': tokens}


def get_dataloaders(model_name: str, config: ConfigValle, split: Literal['train', 'val']):
    dataset = load_dataset(config.dataset, split=split, trust_remote_code=True)
    valle_dataset = ValleDataset(dataset, config)
    dataloader = DataLoader(
        valle_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=get_collate(model_name)(config),
        shuffle=split == 'train',
    )
    return dataloader
