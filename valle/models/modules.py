import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..config import ConfigValle


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
    """Inject some information about the relative or \
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
        """Inputs of forward function
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
        *,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Multi-Head Attention Forward Pass with kv-cache support

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
            out: Output tensor of shape ``(batch_size, seq_len, d_model)``
            kv: Tuple of key-value cache tensors of shape \
                ``(batch_size, n_heads, seq_len, d_model)``
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

        # apply attention mask
        if attn_mask is not None:
            attn_mask = self.merge_masks(batch_size, attn_mask, padding_mask)
            assert attn_mask is not None, 'attn_mask should not be None'
            # Scale dot product attention takes `False` as mask
            attn_mask = ~attn_mask.to(dtype=torch.bool)

        # scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # pylint: disable=not-callable

        # combine heads
        out = rearrange(attn, 'b h n d -> b n (h d)')
        out = self.out(out)

        return out, kv

    def merge_masks(
        self, batch_size: int, attn_mask: torch.Tensor | None, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Merge attention and padding masks into a single mask

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

    def __init__(self, config: ConfigValle) -> None:
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.ffn = FeedForward(
            self.config.d_model, self.config.dim_feedforward, dropout=self.config.dropout
        )
        self.norm1 = self._get_norm()(self.config.d_model)
        self.norm2 = self._get_norm()(self.config.d_model)
        self.dropout1 = nn.Dropout(self.config.dropout)
        self.dropout2 = nn.Dropout(self.config.dropout)
        self.activation = self._get_activation()()

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Encoder Layer Forward Pass

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, d_model)``
            padding_mask: Padding mask tensor of shape ``(batch_size, seq_len)``. \
                Defaults to None.
            attn_mask: Attention mask tensor of shape ``(seq_len, seq_len)``. \
                Defaults to None.
            embedding: Embedding tensor of shape ``(batch_size, d_model)``. \
                Defaults to None.
            kv_cache: Key-Value cache tensor of shape ``(batch_size, seq_len, d_model)``. \
                Defaults to None.
            use_cache: Whether to use key-value cache. Defaults to False.

        Returns:
            x: Output tensor of shape ``(batch_size, seq_len, d_model)``
            kv: Tuple of key-value cache tensors of shape \
                ``(batch_size, n_heads, seq_len, d_model)``
        """
        norm_opt = {} if self.config.norm == 'LayerNorm' else {'embedding': embedding}
        x_attn, next_kv_cache = self.self_attn(
            self.norm1(x, **norm_opt),
            attn_mask=attn_mask,
            padding_mask=padding_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + self.dropout1(x_attn)
        x = x + self.dropout2(self.ffn(self.norm2(x, **norm_opt)))

        return x, next_kv_cache

    def _get_norm(self):
        norm_dict = {
            'LayerNorm': nn.LayerNorm,
            'AdaptiveLayerNorm': AdaptiveLayerNorm,
        }
        return norm_dict[self.config.norm]

    def _get_activation(self):
        activation_dict = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
        }
        return activation_dict[self.config.activation]


class Transformer(nn.Module):
    """Transformer Encoder"""

    def __init__(self, hparams: ConfigValle) -> None:
        super().__init__()
        self.hparams = hparams
        self.layers = nn.ModuleList([EncoderLayer(hparams) for _ in range(self.hparams.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        kv_cache: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """Transformer Encoder Forward Pass

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, d_model)``
            padding_mask: Padding mask tensor of shape ``(batch_size, seq_len)``. \
                Defaults to None.
            attn_mask: Attention mask tensor of shape ``(seq_len, seq_len)``. \
                Defaults to None.
            embedding: Embedding tensor of shape ``(batch_size, d_model)``. \
                Defaults to None.
            kv_cache: Key-Value cache tuple of tensors of shape \
                ``(batch_size, n_heads, seq_len, d_model)``. Defaults to None.
            use_cache: Whether to use key-value cache. Defaults to False.
            return_attn_weights: Whether to return attention weights. Defaults to False.

        Returns:
            x: Output tensor of shape ``(batch_size, seq_len, d_model)``
            new_kv: Tuple of key-value cache tensors of shape \
                ``(batch_size, n_heads, seq_len, d_model)``
        """
        new_kv: tuple = ()
        if use_cache and kv_cache is not None:
            x = x[:, -1:]
            attn_mask = None
        else:
            kv_cache = tuple([None] * self.hparams.num_layers)
        for layer, past_kv in zip(self.layers, kv_cache, strict=False):
            x, next_kv_cache = layer(
                x,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
                embedding=embedding,
                kv_cache=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv = new_kv + (next_kv_cache,)
        return x, new_kv


class SummaryMixin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
