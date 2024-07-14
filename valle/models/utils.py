import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.generation.utils import top_k_top_p_filtering


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


def topk_sampling(
    logits: torch.Tensor, top_k: int = 50, tok_p: float = 1.0, temperature: float | None = 1.0
):
    """Top-k sampling.

    Args:
        logits: Logits tensor (b c t)
        top_k: Top-k value
        tok_p: Token probability
        temperature: Temperature

    Returns:
        Sampled tokens tensor (b t)"""
    if temperature is not None:
        logits = logits / temperature

    # Sampling
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=tok_p)
    sampled_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    logprobs = F.log_softmax(logits, dim=-1)
    current_logprobs = torch.gather(logprobs, -1, sampled_token)

    return sampled_token, current_logprobs
