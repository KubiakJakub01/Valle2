from typing import Literal

import torch
from datasets import load_dataset
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import ConfigValle
from .models import EncodecPip
from .utils import normalize_audio


class ValleDataset(Dataset):
    def __init__(self, dataset, config: ConfigValle):
        self.dataset = dataset
        self.config = config
        self.encodec_pip = EncodecPip()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load item
        item = self.dataset[idx]
        audio = rearrange(torch.tensor(item['audio']['array'], dtype=torch.float32), 't -> 1 t')
        sr = item['audio']['sampling_rate']
        tokens = torch.tensor(item['text_token'])

        # Normalize audio
        audio = rearrange(normalize_audio(audio, sr, self.encodec_pip.sampling_rate), '1 t -> t')

        # Encode audio
        codes = self.encodec_pip.encode(audio)

        return {'codes': codes, 'tokens': tokens}


def get_dataloaders(hparams: ConfigValle, split: Literal['train', 'val']):
    dataset = load_dataset(hparams.dataset, split=split)
    valle_dataset = ValleDataset(dataset, hparams)
    dataloader = DataLoader(valle_dataset, batch_size=hparams.batch_size, shuffle=True)
    return dataloader


def collate_list(x_list: list[Tensor]) -> tuple[Tensor, Tensor]:
    x_lens = torch.tensor(list(map(len, x_list)), dtype=torch.int64)
    x = pad_sequence(x_list, batch_first=True)
    return x, x_lens
