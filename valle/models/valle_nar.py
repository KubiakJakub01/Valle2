import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from .modules import AdaptiveLayerNorm, PositionalEncoding


class ValleNAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

        self.eos_token = hparams.num_audio_tokens
        self.bos_token = hparams.num_audio_tokens + 1

        # Embeddings
        self.tokens_emb = nn.Embedding(hparams.vocab_size, hparams.d_model)
        self.audio_embs = nn.ModuleList(
            [
                nn.Embedding(hparams.num_audio_tokens, hparams.d_model)
                for _ in range(hparams.num_quantizers)
            ]
        )
        self.tokens_position_emb = PositionalEncoding(hparams.d_model)
        self.audio_position_emb = PositionalEncoding(hparams.d_model)
        self.stage_embs = nn.ModuleList(
            [
                nn.Embedding(hparams.num_audio_tokens, hparams.d_model)
                for _ in range(hparams.num_quantizers - 1)
            ]
        )

        # Decoder
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hparams.d_model,
                nhead=hparams.n_head,
                dim_feedforward=hparams.dim_feedforward * 4,
                dropout=hparams.dropout,
                activation=hparams.activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=hparams.num_layers,
            norm=AdaptiveLayerNorm(d_model=hparams.d_model, norm=nn.LayerNorm(hparams.d_model)),
        )

        # Project to output
        self.proj = nn.Linear(hparams.d_model, hparams.num_audio_tokens + 1, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self, tokens_list: list[torch.Tensor], codes_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            tokens_list: List of token sequences (token_len).
            codes_list: List of audio codes (codes_len, quantization_layers).

        Returns:
            loss: Loss value.
        """
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        x_lens = list(map(len, tokens_list))
        x = pad_sequence(tokens_list, batch_first=True)
        x = self.tokens_emb(x)  # (b t c)
        x = self.tokens_position_emb(x)

        # Prepare prompt and target audio
        # Draw a random quantization layer
        layer = random.randint(1, self.hparams.num_quantizers - 1)
        codes = pad_sequence(codes_list, batch_first=True)
        y_emb, prefix_len = self._prepare_audio_codes(codes, layer)
        y_emb = self.audio_position_emb(y_emb)

        # Prepare target audio
        target = codes[:, prefix_len:, layer]

        # Concatenate tokens and codes
        xy = torch.cat([x, y_emb], dim=1)

        # Forward pass
        z = self.decoder((xy, self.stage_embs[layer - 1].weight))
        z = z[:, max(x_lens) + prefix_len]

        # Project to output
        logits = self.proj(z)

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss

    def _prepare_audio_codes(self, codes: torch.Tensor, nar_stage: int) -> tuple[torch.Tensor, int]:
        """Prepare prompt audio.

        Args:
            codes: Audio codes (batch_size, codes_len, quantization_layers).

        Returns:
            y_emb: Prompt audio embeddings (batch_size, codes_len, d_model).
            prefix_len: Length of the prompt audio.
        """
        # Cut 3 seconds of audio or 1/3 of the audio
        codes_len = codes.shape[-1]
        prefix_len = min(codes_len // 3, 3 * self.hparams.quantization_factor)
        prompts_codes = self.audio_embs[0](codes[:, :prefix_len, 0])
        emb_codes = self.audio_embs[0](codes[:, prefix_len:, 0])
        for j in range(1, self.hparams.num_quantizers):
            prompts_codes += self.audio_embs[j](codes[:, :prefix_len, j])
            if j < nar_stage:
                emb_codes += self.audio_embs[j](codes[:, prefix_len:, j])
        y_emb = torch.concat([prompts_codes, emb_codes], axis=1)

        return y_emb, prefix_len
