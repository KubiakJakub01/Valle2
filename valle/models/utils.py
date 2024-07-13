import torch
from einops import rearrange


def create_pad_mask(x_len_list, device):
    """1 is valid region and 0 is masked region."""
    seq = rearrange(torch.arange(max(x_len_list), device=device), 't -> 1 t')
    stop = rearrange(torch.tensor(x_len_list, device=device), 'b -> b 1')
    return (seq >= stop).bool()


def build_attn_mask(x_len: int, y_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask for ValleAR model.

    1 - Masked, 0 - Not masked

    Args:
        x_len: Length of tokens
        y_len: Length of audio codes
        device: Device

    Returns:
        Attention mask tensor (x_len + y_len, x_len + y_len)"""
    x_mask = torch.cat(
        (
            torch.zeros((x_len, x_len), dtype=torch.bool, device=device),
            torch.ones((x_len, y_len), dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    y_mask = torch.cat(
        (
            torch.zeros((y_len, x_len), dtype=torch.bool, device=device),
            torch.triu(torch.ones((y_len, y_len), dtype=torch.bool, device=device), diagonal=1),
        ),
        dim=1,
    )
    return torch.cat((x_mask, y_mask), dim=0)
