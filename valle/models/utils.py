import torch
from einops import rearrange


def create_pad_mask(x_len_list, device):
    """1 is valid region and 0 is masked region."""
    seq = rearrange(torch.arange(max(x_len_list), device=device), 't -> 1 t')
    stop = rearrange(torch.tensor(x_len_list, device=device), 'b -> b 1')
    return (seq >= stop).bool()
