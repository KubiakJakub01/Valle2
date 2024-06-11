import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from .modules import PositionalEncoding, TokenEmbedding, Transformer
from .utils import create_pad_mask


class ValleNAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

        self.eos_token = hparams.num_audio_tokens
        self.bos_token = hparams.num_audio_tokens + 1

        # Embeddings
        self.tokens_emb = TokenEmbedding(hparams.vocab_size, hparams.d_model)
        self.audio_embs = nn.ModuleList(
            [
                TokenEmbedding(hparams.num_audio_tokens, hparams.d_model)
                for _ in range(hparams.num_quantizers)
            ]
        )
        self.tokens_position_emb = PositionalEncoding(hparams.d_model)
        self.audio_position_emb = PositionalEncoding(hparams.d_model)
        self.stage_embs = nn.ModuleList(
            [
                TokenEmbedding(hparams.num_audio_tokens, hparams.d_model)
                for _ in range(hparams.num_quantizers - 1)
            ]
        )

        # Decoder
        self.transformer = Transformer(hparams)

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
            tokens_list: List of token sequences (tokens_len).
            codes_list: List of audio codes (codes_len, quantization_layers).

        Returns:
            loss: Loss value.
        """
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        tokens_lens = list(map(len, tokens_list))
        tokens_len = max(tokens_lens)
        tokens = pad_sequence(tokens_list, batch_first=True)
        tokens = self.tokens_emb(tokens)  # (b t c)
        tokens = self.tokens_position_emb(tokens)

        # Prepare prompt and target audio
        codes_lens = list(map(len, codes_list))
        layer = random.randint(1, self.hparams.num_quantizers - 1)
        codes = pad_sequence(codes_list, batch_first=True)
        codes, prefix_len = self._prepare_audio_codes(codes, layer)
        codes = self.audio_position_emb(codes)

        # Prepare target audio
        target = codes[:, prefix_len:, layer]

        # Prepare mask
        codes_pad_mask = F.pad(
            create_pad_mask(codes_lens, self.device),
            (tokens_len, 0),
            value=False,
        )  # [tokens_len, codes_len]

        # Concatenate tokens and codes
        xy = torch.cat([tokens, codes], dim=1)

        # Forward pass
        z = self.transformer(
            xy, padding_mask=codes_pad_mask, embedding=self.stage_embs[layer - 1].weight
        )
        z = z[:, max(tokens_lens) + prefix_len]

        # Project to output
        logits = self.proj(z)

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Generate audio from tokens.

        Args:
            tokens: Token sequence (tokens_len).
            codes: Audio codes (codes_len, quantization_layers).

        Returns:
            audio: Generated audio (codes_len).
        """
        # Prepare tokens
        tokens = rearrange(tokens, 't -> 1 t')
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Prepare audio
        codes, _ = self._prepare_audio_codes(codes, self.hparams.num_quantizers)

        # Prepare mask
        tokens_len = tokens.shape[1]
        codes_len = codes.shape[1]
        codes_pad_mask = F.pad(
            create_pad_mask([codes_len], self.device),
            (tokens_len, 0),
            value=False,
        )

        # Concatenate tokens and codes
        xy = torch.cat([tokens, codes], dim=1)

        # Forward pass
        z = self.transformer(xy, padding_mask=codes_pad_mask, embedding=self.stage_embs[-1].weight)

        # Project to output
        logits = self.proj(z)

        # Generate audio
        audio = torch.argmax(logits, dim=-1)

        return audio

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
        prompts_codes: torch.Tensor = self.audio_embs[0](codes[:, :prefix_len, 0])
        emb_codes: torch.Tensor = self.audio_embs[0](codes[:, prefix_len:, 0])
        for j in range(1, self.hparams.num_quantizers):
            prompts_codes += self.audio_embs[j](codes[:, :prefix_len, j])
            if j < nar_stage:
                emb_codes += self.audio_embs[j](codes[:, prefix_len:, j])
        y_emb = torch.concat((prompts_codes, emb_codes), dim=1)

        return y_emb, prefix_len
