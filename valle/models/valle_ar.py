import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import optim

from ..config import ConfigValle
from ..utils import to_device
from .modules import PositionalEncoding, TokenEmbedding, Transformer
from .utils import build_attn_mask, build_pad_mask, get_best_beam, topk_sampling


class ValleAR(L.LightningModule):
    def __init__(self, config: ConfigValle):
        super().__init__()
        self.config = config

        # Embeddings
        self.tokens_emb = TokenEmbedding(self.config.vocab_size, self.config.d_model)
        self.audio_emb = TokenEmbedding(self.config.num_audio_tokens + 2, self.config.d_model)
        self.tokens_position_emb = PositionalEncoding(self.config.d_model)
        self.audio_position_emb = PositionalEncoding(self.config.d_model)

        # Transformer
        self.transformer = Transformer(self.config)

        # Project to output
        self.proj = nn.Linear(self.config.d_model, self.config.num_audio_tokens + 1, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def eos_token(self):
        return self.config.num_audio_tokens

    @property
    def bos_token(self):
        return self.config.num_audio_tokens + 1

    def training_step(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Batch data

        Returns:
            loss: Loss value
        """
        # pylint: disable=arguments-differ
        batch = to_device(batch, self.device)
        codes = batch['codes']
        codes_lens = batch['codes_lens']
        tokens = batch['tokens']
        tokens_lens = batch['tokens_lens']
        target = batch['target']

        # Prepare tokens
        tokens = self.tokens_emb(tokens)  # (b t c)
        tokens = self.tokens_position_emb(tokens)

        # Prepare audio
        codes = self.audio_emb(codes)  # (b t c)
        codes = self.audio_position_emb(codes)

        # Decoder
        padding_mask = F.pad(
            build_pad_mask(codes_lens, self.device),
            (max(tokens_lens), 0),
            value=False,
        )
        attn_mask = build_attn_mask(max(tokens_lens), max(codes_lens), self.device)
        transformer_output, *_ = self.transformer(
            torch.cat((tokens, codes), dim=1),
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        transformer_output = transformer_output[:, max(tokens_lens) :]

        # Project to output
        logits = rearrange(self.proj(transformer_output), 'b t c -> b c t')

        # Compute loss
        loss = F.cross_entropy(logits, target)

        self.log('train/loss', loss)

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
            output_codes: Output audio codes (output_len)
        """
        assert prompt_tokens.dim() == 1, 'Prompt tokens should be 1D tensor.'
        assert prompt_codes.dim() == 2, 'Prompt codes should be 2D tensor.'
        if target_tokens is not None:
            assert target_tokens.dim() == 1, 'Target tokens should be 1D tensor.'

        # Get first layer from prompt codes and add bos token
        prompt_codes = rearrange(
            F.pad(prompt_codes[..., 0], (1, 0), value=self.bos_token), 't -> 1 t'
        )  # (b t)
        prompt_len = prompt_codes.shape[1]

        # Prepare tokens
        tokens = (
            torch.cat((prompt_tokens, target_tokens), dim=0)
            if target_tokens is not None
            else prompt_tokens
        )
        tokens_len = tokens.shape[0]
        tokens = rearrange(tokens, 't -> 1 t')  # (b t)
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Attention mask
        attn_mask = build_attn_mask(tokens_len, prompt_len, self.device)

        # Prepare decoding variables
        kv_cache = None
        sum_logprobs = torch.zeros(self.config.num_beams, device=self.device)
        tokens = tokens.repeat(self.config.num_beams, 1, 1)
        prompt_codes = prompt_codes.repeat(self.config.num_beams, 1)

        # Decoding loop
        for _ in range(self.config.max_audio_len):
            # Prepare audio codes
            codes = self.audio_emb(prompt_codes)
            codes = self.audio_position_emb(codes)

            # Concatenate tokens and codes
            transformer_input = torch.cat([tokens, codes], dim=1)

            # Transformer
            transformer_output, kv_cache = self.transformer(
                transformer_input,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                use_cache=self.config.use_kv_cache,
            )

            # Project to output
            logits = self.proj(transformer_output)[:, -1]

            # Sampling
            samples, current_logprobs = topk_sampling(
                logits=logits,
                top_k=self.config.top_k,
                tok_p=self.config.tok_p,
                temperature=self.config.temperature,
            )
            sum_logprobs += current_logprobs * (prompt_codes[:, -1] != self.eos_token)
            samples[prompt_codes[:, -1] == self.eos_token] = self.eos_token
            if (samples[:, -1] == self.eos_token).all():
                break
            prompt_codes = torch.cat([prompt_codes, samples], dim=1)

        # Prepare output
        output_codes = get_best_beam(
            prompt_codes, sum_logprobs, self.eos_token, self.config.length_penalty
        )
        output_codes = output_codes[prompt_len:]
        output_codes = output_codes[output_codes != self.eos_token]

        return output_codes

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            fused=True,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            self.config.lr_warmup,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
