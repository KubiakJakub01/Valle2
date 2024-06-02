import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from .modules import PositionalEncoding, TokenEmbedding


class ValleAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

        self.eos_token = hparams.num_audio_tokens
        self.bos_token = hparams.num_audio_tokens + 1

        # Embeddings
        self.tokens_emb = TokenEmbedding(hparams.vocab_size, hparams.d_model)
        self.audio_emb = TokenEmbedding(hparams.num_audio_tokens + 2, hparams.d_model)
        self.tokens_position_emb = PositionalEncoding(hparams.d_model)
        self.audio_position_emb = PositionalEncoding(hparams.d_model)

        # Decoder
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hparams.d_model,
                nhead=hparams.n_heads,
                dim_feedforward=hparams.dim_feedforward,
                dropout=hparams.dropout,
                activation=hparams.activation,
                batch_first=True,
            ),
            num_layers=hparams.num_layers,
            norm=nn.LayerNorm(hparams.d_model),
        )

        # Project to output
        self.proj = nn.Linear(hparams.d_model, hparams.num_audio_tokens + 1, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        tokens_list: list[torch.Tensor],
        codes_list: list[torch.Tensor],
    ):
        """Forward pass.

        Args:
            tokens_list: List of tokens tensor (tokens_len)
            codes_list: List of audio codes tensor (1, codes_len)
        """
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        x_len = max(map(len, tokens_list))
        x = pad_sequence(tokens_list, batch_first=True)
        x = self.tokens_emb(x)  # (b t c)
        x = self.tokens_position_emb(x)

        # Prepare audio
        y_len = max(map(len, codes_list)) + 1
        target = pad_sequence(
            [F.pad(codes, (0, 1), value=self.eos_token) for codes in codes_list], batch_first=True
        )
        y = pad_sequence(
            [F.pad(codes, (1, 0), value=self.bos_token) for codes in codes_list], batch_first=True
        )
        y = self.audio_emb(y)  # (b t c)
        y = self.audio_position_emb(y)

        # Prepare mask
        x_mask = torch.cat(
            (
                torch.zeros((x_len, x_len), dtype=torch.bool, device=self.device),
                torch.ones((x_len, y_len), dtype=torch.bool, device=self.device),
            ),
            dim=1,
        )
        y_mask = torch.cat(
            (
                torch.zeros((y_len, x_len), dtype=torch.bool, device=self.device),
                torch.triu(
                    torch.ones((y_len, y_len), dtype=torch.bool, device=self.device), diagonal=1
                ),
            ),
            dim=1,
        )
        mask = torch.cat((x_mask, y_mask), dim=0)

        # Decoder
        xy = torch.cat((x, y), dim=1)
        z = self.decoder(xy, mask=mask)
        z = z[:, x_len:]

        # Project to output
        logits = rearrange(self.proj(z), 'b t c -> b c t')

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss
