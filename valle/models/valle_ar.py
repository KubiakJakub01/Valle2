import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from .modules import PositionalEncoding, TokenEmbedding, Transformer


class ValleAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

        self.eos_token = self.hparams.num_audio_tokens
        self.bos_token = self.hparams.num_audio_tokens + 1

        # Embeddings
        self.tokens_emb = TokenEmbedding(self.hparams.vocab_size, self.hparams.d_model)
        self.audio_emb = TokenEmbedding(self.hparams.num_audio_tokens + 2, self.hparams.d_model)
        self.tokens_position_emb = PositionalEncoding(self.hparams.d_model)
        self.audio_position_emb = PositionalEncoding(self.hparams.d_model)

        # Transformer
        self.transformer = Transformer(self.hparams)

        # Project to output
        self.proj = nn.Linear(self.hparams.d_model, self.hparams.num_audio_tokens + 1, bias=False)

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

        # Decoder
        xy = torch.cat((x, y), dim=1)
        z, *_ = self.transformer(xy, attn_mask=self.build_attn_mask(x_len, y_len))
        z = z[:, x_len:]

        # Project to output
        logits = rearrange(self.proj(z), 'b t c -> b c t')

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        codes: torch.Tensor,
    ):
        """Generate audio codes.

        Args:
            tokens: Tokens tensor (tokens_len)
            codes: Audio codes tensor (quantization_layers, codes_len)

        Returns:
            Generated audio codes tensor (1, codes_len)
        """
        x = self.tokens_emb(tokens)

        y = F.pad(codes, (1, 0), value=self.bos_token)
        y = self.audio_emb(y)
        y = self.audio_position_emb(y)

        for _ in range(self.hparams.max_audio_len):
            xy = torch.cat((x, y), dim=1)
            z, *_ = self.transformer(xy)
            z = z[:, -1:]

            logits = self.proj(z)
            y = torch.cat((y, logits.argmax(dim=-1)), dim=1)

            if y[0, -1] == self.eos_token:
                break

        return y[:, 1:]

    def build_attn_mask(self, x_len: int, y_len: int) -> torch.Tensor:
        """Prepare attention mask.

        1 - Masked, 0 - Not masked

        Args:
            x_len: Length of tokens
            y_len: Length of audio codes

        Returns:
            Attention mask tensor (x_len + y_len, x_len + y_len)"""
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
        return torch.cat((x_mask, y_mask), dim=0)
