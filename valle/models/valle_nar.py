import random

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from ..utils import log_info
from .modules import AdaptiveLayerNorm, PositionalEncoding


class ValleAR(nn.Module):
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
            codes_list: List of audio codes (quantization_layers, codes_len).

        Returns:
            loss: Loss value.
        """
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        x = pad_sequence(tokens_list, batch_first=True)
        x = self.tokens_emb(x)  # (b t c)
        x = self.tokens_position_emb(x)

        # Prepare prompt and target audio
        # Draw a random quantization layer
        layer = random.randint(1, self.hparams.num_quantizers - 1)
        codes_layer = rearrange(
            pad_sequence([codes[layer] for codes in codes_list], batch_first=True), 'b 1 t -> b t'
        )
        prompt_codes, target_codes = self._prepare_promt_audio(codes_layer)

        log_info(f'Prompt codes shape: {prompt_codes.shape}')
        log_info(f'Target codes shape: {target_codes.shape}')

    def _prepare_promt_audio(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare prompt audio.

        Args:
            codes: Audio codes (batch_size, codes_len).

        Returns:
            x: Prompt audio (b t c).
            x_pos: Prompt audio position (b t c).
        """
        # Cut 3 seconds of audio or 1/3 of the audio
        _, codes_len = codes.shape
        start = min(codes_len // 3, 3 * self.hparams.quantization_factor)
        prompt_codes = codes[:, :start]
        target_codes = codes[:, start:]

        return prompt_codes, target_codes
