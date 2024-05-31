import math

import torch
import torch.nn as nn
from einops import rearrange

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


class FeedForward(nn.Module):
    """Feed Forward Neural Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class EncoderLayer(nn.Module):
    """Encoder Layer"""

    def __init__(self, hparams: ValleHparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.self_attn = nn.MultiheadAttention(
            self.hparams.d_model, self.hparams.n_head, dropout=self.hparams.dropout
        )
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
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_opt = {} if self.hparams.norm == 'LayerNorm' else {'embedding': embedding}
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src, **norm_opt)

        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src, **norm_opt)
        return src

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
