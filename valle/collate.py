from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .config import ConfigValle


def get_collate(model_name: str):
    collate_dict = {
        'valle_ar': ValleARCollate,
        'valle_nar': ValleNARCollate,
    }
    return collate_dict[model_name]


@dataclass
class ValleARCollate:
    config: ConfigValle

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        codes_list = []
        targets_list = []
        tokens_list = []
        for item in batch:
            codes_ = item['codes'][0]  # Only first layer
            codes = F.pad(codes_, (1, 0), value=self.config.bos_token)
            target = F.pad(codes_, (0, 1), value=self.config.eos_token)
            codes_list.append(codes)
            targets_list.append(target)
            tokens_list.append(item['tokens'])
        codes, codes_lens = collate_1d(codes_list)
        target, _ = collate_1d(targets_list)
        tokens, tokens_lens = collate_1d(tokens_list)
        assert (codes_lens > tokens_lens).all(), 'Codes length must be greater than tokens length.'
        return {
            'codes': codes,
            'codes_lens': codes_lens,
            'target': target,
            'tokens': tokens,
            'tokens_lens': tokens_lens,
        }


@dataclass
class ValleNARCollate:
    config: ConfigValle

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        codes, codes_lens = collate_2d_sequences([item['codes'] for item in batch])
        tokens, tokens_lens = collate_1d([item['tokens'] for item in batch])
        assert (codes_lens > tokens_lens).all(), 'Codes length must be greater than tokens length.'
        return {
            'codes': codes,
            'codes_lens': codes_lens,
            'tokens': tokens,
            'tokens_lens': tokens_lens,
        }


def collate_1d(x_list: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Collate list of tensors.

    Args:
        x_list: List of tensors.

    Returns:
        x: Padded tensor.
        x_lens: Lengths of tensors.
    """
    x_lens = torch.tensor(list(map(len, x_list)), dtype=torch.int64)
    x = pad_sequence(x_list, batch_first=True)
    return x, x_lens


def collate_2d_sequences(
    sequences: list[torch.Tensor],
    padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate list of 2D sequences.

    Args:
        sequences: List of 2D tensors.
        padding: Padding value.

    Returns:
        Padded tensor and lengths of sequences.
    """
    sequence_lens = torch.tensor([s.size(1) for s in sequences], dtype=torch.int64)
    max_len = int(sequence_lens.max())
    sequences_ = []
    padding_tensor = torch.tensor(padding, dtype=sequences[0].dtype)
    for sequence in sequences:
        sequence_ = torch.zeros((sequence.size(0), max_len), dtype=sequence.dtype)
        sequence_[:, : sequence.size(1)] = sequence
        sequence_[:, sequence.size(1) :] = padding_tensor
        sequences_.append(sequence_)
    return torch.stack(sequences_), sequence_lens
