import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
from .modules import PositionalEncoding, TokenEmbedding, Transformer
from .utils import build_attn_mask, create_pad_mask, get_best_beam, topk_sampling


class ValleAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

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

    @property
    def eos_token(self):
        return self.hparams.num_audio_tokens

    @property
    def bos_token(self):
        return self.hparams.num_audio_tokens + 1

    def forward(
        self,
        tokens_list: list[torch.Tensor],
        codes_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens_list: List of tokens tensor (tokens_len)
            codes_list: List of audio codes tensor (1, codes_len)

        Returns:
            loss: Loss value
        """
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        tokens_lens = list(map(len, tokens_list))
        tokens_len = max(tokens_lens)
        tokens = pad_sequence(tokens_list, batch_first=True)
        tokens = self.tokens_emb(tokens)  # (b t c)
        tokens = self.tokens_position_emb(tokens)

        # Prepare audio
        codes_lens = [x + 1 for x in map(len, codes_list)]
        codes_len = max(codes_lens)
        target = pad_sequence(
            [F.pad(codes, (0, 1), value=self.eos_token) for codes in codes_list], batch_first=True
        )
        codes = pad_sequence(
            [F.pad(codes, (1, 0), value=self.bos_token) for codes in codes_list], batch_first=True
        )
        codes = self.audio_emb(codes)  # (b t c)
        codes = self.audio_position_emb(codes)

        # Decoder
        transformer_output, *_ = self.transformer(
            torch.cat((tokens, codes), dim=1),
            padding_mask=create_pad_mask(codes_lens, self.device),
            attn_mask=build_attn_mask(tokens_len, codes_len, self.device),
        )
        transformer_output = transformer_output[:, tokens_len:]

        # Project to output
        logits = rearrange(self.proj(transformer_output), 'b t c -> b c t')

        # Compute loss
        loss = F.cross_entropy(logits, target)

        return loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        prompt_codes: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ):
        """Generate first layer audio codes from tokens.

        Args:
            prompt_tokens: Prompt tokens (prompt_tokens_len)
            prompt_codes: Prompt audio codes (codes_len, num_quantizers)
            target_tokens: Target tokens (target_tokens_len)

        Returns:
            output_codes: Output audio codes (output_len, 1)
        """
        assert prompt_tokens.dim() == 1, 'Prompt tokens should be 1D tensor.'
        assert prompt_codes.dim() == 2, 'Prompt codes should be 2D tensor.'
        if target_tokens is not None:
            assert target_tokens.dim() == 1, 'Target tokens should be 1D tensor.'

        # Get first layer from prompt codes and add bos token
        codes = rearrange(F.pad(prompt_codes[..., 0], (1, 0), value=self.bos_token), 't 1 -> 1 t 1')
        prompt_len = codes.shape[1]

        # Prepare tokens
        tokens = (
            torch.cat((prompt_tokens, target_tokens), dim=0)
            if target_tokens is not None
            else prompt_tokens
        )
        tokens_len = tokens.shape[0]
        tokens = rearrange(tokens, 't -> 1 t')
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Attention mask
        attn_mask = build_attn_mask(tokens_len, prompt_len, self.device)

        # Prepare decoding variables
        kv_cache = None
        sum_logprobs = torch.zeroes(self.hparams.num_beams, device=self.device)
        tokens = tokens.repeat(self.hparams.num_beams, 1, 1)
        codes = codes.repeat(self.hparams.num_beams, 1)

        # Decoding loop
        for _ in range(self.hparams.max_audio_len):
            # Prepare audio codes
            codes = self.audio_emb(codes)
            codes = self.audio_position_emb(codes)

            # Concatenate tokens and codes
            transformer_input = torch.cat([tokens, codes], dim=1)

            # Transformer
            transformer_output, kv_cache = self.transformer(
                transformer_input,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                use_kv_cache=self.hparams.use_kv_cache,
            )

            # Project to output
            logits = self.proj(transformer_output)[:, -1]

            # Sampling
            samples, current_logprobs = topk_sampling(
                logits=logits,
                top_k=self.hparams.top_k,
                tok_p=self.hparams.tok_p,
                temperature=self.hparams.temperature,
            )
            sum_logprobs += current_logprobs * (codes[:, -1] != self.eos_token).float()
            samples[codes[:, -1] == self.eos_token] = self.eos_token
            if (samples[:, -1] == self.eos_token).all():
                break
            codes = torch.cat([codes, samples], dim=1)

        # Prepare output
        output_codes = get_best_beam(
            codes, sum_logprobs, self.eos_token, self.hparams.length_penalty
        )
        output_codes = output_codes[prompt_len:]
        output_codes = output_codes[output_codes != self.eos_token]

        return output_codes
