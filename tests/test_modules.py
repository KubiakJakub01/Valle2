import pytest
import torch

from valle.models.modules import MultiHeadAttention


@pytest.mark.parametrize(
    'd_model, n_heads, batch_size, seq_len',
    [
        (512, 8, 4, 5),
        (256, 4, 8, 10),
        (128, 2, 16, 20),
    ],
)
def test_multi_head_attention(
    d_model: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
):
    attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    head_dim = d_model // n_heads
    assert attention.head_dim == head_dim
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    output, attn, kv = attention(x, attn_mask=mask, use_cache=True)
    k, v = kv
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn.shape == (batch_size, n_heads, seq_len, seq_len)
    assert k.shape == (batch_size, n_heads, seq_len, head_dim)
    assert v.shape == (batch_size, n_heads, seq_len, head_dim)


@pytest.mark.parametrize(
    'd_model, n_heads, batch_size, seq_len, padding_mask, expected_outputs',
    [
        (
            512,
            8,
            4,
            5,
            torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]),
            [120, 112, 96, 72],
        ),
        (
            256,
            4,
            8,
            10,
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            [220, 216, 208, 196, 180, 160, 136, 108],
        ),
    ],
)
def test_merge_masks(
    d_model: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
    padding_mask: torch.Tensor,
    expected_outputs: list[int],
):
    attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = attention.merge_masks(batch_size, attn_mask, padding_mask)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (batch_size, n_heads, seq_len, seq_len)
    for i, expected_output in enumerate(expected_outputs):
        assert (mask[i] == 0.0).sum().item() == expected_output
