import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..hparams import ValleHparams


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or \
            absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, \
            so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = rearrange(x, 'b t c -> t b c')
        x = x + self.pe[: x.size(0), :]
        return rearrange(self.dropout(x), 't b c -> b t c')


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, d_model) -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(x) + bias


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()

        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        r"""Multi-Head Attention Forward Pass with kv-cache support

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, d_model)``
            attn_mask: Attention mask tensor of shape ``(seq_len, seq_len)``. \
                Defaults to None.
            padding_mask: Padding mask tensor of shape ``(batch_size, seq_len)``. \
                Defaults to None.
            kv_cache: Key-Value cache tensor of shape ``(seq_len, batch_size, d_model)``. \
                Defaults to None.
            use_cache: Whether to use key-value cache. Defaults to False.

        Returns:
            x: Output tensor of shape (batch_size, seq_len, d_model)
            attn: Attention tensor of shape (batch_size, n_heads, seq_len, seq_len)
            kv: Key-Value cache tensor of shape (seq_len, batch_size, d_model)
        """
        batch_size = x.shape[0]

        # q, k, v: (batch_size, n_heads, seq_len, head_dim)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # manage key-value cache
        kv = None
        if use_cache and kv_cache is not None:
            past_k = kv_cache[0]
            past_v = kv_cache[1]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        if use_cache:
            kv = (k, v)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # apply attention mask
        if attn_mask is not None:
            merged_mask = self.merge_masks(batch_size, attn_mask, padding_mask)
            attn = attn.masked_fill(merged_mask == 0, float('-inf'))

        # softmax and weighted sum
        attn = torch.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)

        return x, attn, kv

    def merge_masks(
        self, batch_size: int, attn_mask: torch.Tensor | None, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        r"""
        Merge attention and padding masks into a single mask

        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``
            x: embeddings of shape ``(batch_size, seq_len, embed_dim)``

        Returns:
            merged_mask: merged mask of shape ``(batch_size, num_heads, seq_len, seq_len)`` or None
        """
        merged_mask: torch.Tensor | None = None

        if attn_mask is not None:
            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = rearrange(attn_mask, 'b t t -> b 1 t t')
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = repeat(
                    attn_mask, 'h w -> b n h w', b=batch_size, n=self.n_heads
                )

            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = repeat(
                    key_padding_mask, 'b t -> b n h t', b=batch_size, n=self.n_heads, h=1
                )
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        return merged_mask


class FeedForward(nn.Module):
    """Feed Forward Neural Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class EncoderLayer(nn.Module):
    """Encoder Layer"""

    def __init__(self, hparams: ValleHparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.self_attn = MultiHeadAttention(hparams.d_model, hparams.n_heads)
        self.ffn = FeedForward(
            self.hparams.d_model, self.hparams.dim_feedforward, dropout=self.hparams.dropout
        )
        self.norm1 = self._get_norm()(self.hparams.d_model)
        self.norm2 = self._get_norm()(self.hparams.d_model)
        self.dropout1 = nn.Dropout(self.hparams.dropout)
        self.dropout2 = nn.Dropout(self.hparams.dropout)
        self.activation = self._get_activation()()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Encoder Layer Forward Pass
        
        Args:
            x: Input tensor of shape ``(batch_size, seq_len, d_model)``
            padding_mask: Padding mask tensor of shape ``(batch_size, seq_len)``. \
                Defaults to None.
            attn_mask: Attention mask tensor of shape ``(seq_len, seq_len)``. \
                Defaults to None.
            embedding: Embedding tensor of shape ``(batch_size, d_model)``. \
                Defaults to None.
            kv_cache: Key-Value cache tensor of shape ``(seq_len, batch_size, d_model)``. \
                Defaults to None.
            use_cache: Whether to use key-value cache. Defaults to False."""
        norm_opt = {} if self.hparams.norm == 'LayerNorm' else {'embedding': embedding}
        x_attn, attn_weights, kv_cache = self.self_attn(
            self.norm1(x, **norm_opt), attn_mask, padding_mask, kv_cache, use_cache
        )
        x = x + self.dropout1(x_attn)
        x = x + self.dropout2(self.ffn(self.norm2(x, **norm_opt)))

        return x, attn_weights, kv_cache

    def _get_norm(self):
        norm_dict = {
            'LayerNorm': nn.LayerNorm,
            'AdaptiveLayerNorm': AdaptiveLayerNorm,
        }
        return norm_dict[self.hparams.norm]

    def _get_activation(self):
        activation_dict = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
        }
        return activation_dict[self.hparams.activation]


class Encoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, hparams: ValleHparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.layers = nn.ModuleList([EncoderLayer(hparams) for _ in range(hparams.num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask, embedding)
        return src
