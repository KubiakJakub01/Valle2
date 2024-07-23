import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from ..config import ConfigValle
from .modules import PositionalEncoding, TokenEmbedding, Transformer
from .utils import create_pad_mask


class ValleNAR(nn.Module):
    def __init__(self, config: ConfigValle):
        super().__init__()
        self.hparams = config

        self.eos_token = config.num_audio_tokens
        self.bos_token = config.num_audio_tokens + 1

        # Embeddings
        self.tokens_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.codes_embs = nn.ModuleList(
            [
                TokenEmbedding(config.num_audio_tokens, config.d_model)
                for _ in range(config.num_quantizers)
            ]
        )
        self.tokens_position_emb = PositionalEncoding(config.d_model)
        self.audio_position_emb = PositionalEncoding(config.d_model)
        self.stage_embs = nn.ModuleList(
            [TokenEmbedding(1, config.d_model) for _ in range(config.num_quantizers - 1)]
        )

        # Decoder
        self.transformer = Transformer(config)

        # Project to output
        self.proj_layers = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.num_audio_tokens, bias=False)
                for _ in range(config.num_quantizers - 1)
            ]
        )

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
        z, _ = self.transformer(
            xy, padding_mask=codes_pad_mask, embedding=self.stage_embs[layer - 1].weight
        )
        z = z[:, tokens_len + prefix_len]

        # Project to output
        logits = self.proj_layers[layer - 1](z)

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        prompt_codes: torch.Tensor,
        target_tokens: torch.Tensor,
        target_codes_first_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Generate remaining audio codes layers.

        Args:
            prompt_tokens: Token sequences (prompt_tokens_len).
            prompt_codes: Audio codes (prompt_codes_len, quantization_layers).
            target_tokens: Target token sequences (target_tokens_len).
            target_codes_first_layer: Target audio codes (target_codes_len).

        Returns:
            output_codes: Output audio codes (output_len, quantization_layers).
        """
        # Prepare prompts
        emb_prompt_codes = torch.zeros_like(prompt_codes)
        emb_output_codes = torch.zeros_like(target_codes_first_layer)
        output_codes = target_codes_first_layer
        prompt_len, num_quantizers = prompt_codes.shape
        prompt_codes = rearrange(prompt_codes, 't c -> c t')
        for j in range(num_quantizers):
            emb_prompt_codes += self.codes_embs[j](prompt_codes[j])

        # Prepare tokens
        tokens = rearrange(torch.cat([prompt_tokens, target_tokens], dim=0), 't -> 1 t')
        _, tokens_len = tokens.shape
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Decoding loop
        for n_layer in range(1, num_quantizers):
            # Prepare codes
            emb_output_codes += self.codes_embs[n_layer](output_codes)
            codes = rearrange(
                torch.cat([emb_prompt_codes, emb_output_codes], dim=0), 't c -> 1 t c'
            )
            codes = self.audio_position_emb(codes)

            # Transformer
            transformer_input = torch.cat([tokens, codes], dim=1)
            transformer_output = self.transformer(
                transformer_input, embedding=self.stage_embs[n_layer - 1].weight
            )

            # Project to output
            logits = self.proj_layers[n_layer - 1](transformer_output[:, tokens_len + prompt_len :])

            # Sampling
            sampled_tokens = Categorical(logits=logits / self.hparams.temperature).sample()

            # Update output codes
            output_codes = torch.cat([output_codes, sampled_tokens], dim=0)

        return output_codes

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
        prompts_codes: torch.Tensor = self.codes_embs[0](codes[:, :prefix_len, 0])
        emb_codes: torch.Tensor = self.codes_embs[0](codes[:, prefix_len:, 0])
        for j in range(1, self.hparams.num_quantizers):
            prompts_codes += self.codes_embs[j](codes[:, :prefix_len, j])
            if j < nar_stage:
                emb_codes += self.codes_embs[j](codes[:, prefix_len:, j])
        y_emb = torch.concat((prompts_codes, emb_codes), dim=1)

        return y_emb, prefix_len
