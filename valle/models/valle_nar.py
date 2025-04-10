import random

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import optim
from torch.distributions import Categorical
from torchmetrics.classification import MulticlassAccuracy

from ..config import ConfigValle
from ..utils import to_device
from .encodec_pip import EncodecPip
from .modules import PositionalEncoding, TokenEmbedding, Transformer
from .utils import build_pad_mask


class ValleNAR(L.LightningModule):
    def __init__(self, config: ConfigValle):
        super().__init__()
        self.config = config

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

        # Metrics
        self.accuracy = MulticlassAccuracy(
            self.config.num_audio_tokens,
            average='micro',
            multidim_average='global',
            ignore_index=self.eos_token,
        )

        self.validation_dict: dict[str, torch.Tensor] = {}

    @property
    def device(self):
        return next(self.parameters()).device

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
        tokens = batch['tokens']
        tokens_lens = batch['tokens_lens']

        # Train on random layer
        layer = random.randint(1, self.config.num_quantizers - 1)
        codes_emb, prefix_len = self._prepare_audio_codes(codes, layer)
        target = codes[:, layer, int(prefix_len.max().item()) :]

        # Forward pass
        logits = self.forward(tokens, codes_emb, prefix_len.to(self.device), tokens_lens, layer)

        # Compute loss
        loss = F.cross_entropy(logits, target)
        accuracy = self.accuracy(logits, target)
        self.log('train/loss', loss)
        self.log('train/acc', accuracy)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch: dict[str, torch.Tensor], **kwargs):
        """Validation step.

        Args:
            batch: Batch data

        Returns:
            loss: Loss value
        """
        # pylint: disable=arguments-differ
        batch = to_device(batch, self.device)
        codes = batch['codes']
        tokens = batch['tokens']
        tokens_lens = batch['tokens_lens']

        # Forward pass
        layer = random.randint(1, self.config.num_quantizers - 1)
        codes_emb, prefix_len = self._prepare_audio_codes(codes, layer)
        target = codes[:, layer, int(prefix_len.max().item()) :]

        # Forward pass
        logits = self.forward(tokens, codes_emb, prefix_len.to(self.device), tokens_lens, layer)

        # Compute loss
        loss = F.cross_entropy(logits, target)
        accuracy = self.accuracy(logits, target)

        # Log metrics
        self.log('val/loss', loss)
        self.log('val/acc', accuracy)

        if not self.validation_dict:
            self.validation_dict['codes'] = codes
            self.validation_dict['tokens'] = tokens

    def on_validation_epoch_end(self):
        """On validation epoch end."""
        if not self.validation_dict:
            return

        codes: torch.Tensor = rearrange(self.validation_dict['codes'], '1 q t -> q t')
        tokens: torch.Tensor = rearrange(self.validation_dict['tokens'], '1 t -> t')

        # Make inference
        codes_first_layer = codes[0, :]
        output_codes = self.generate(tokens, codes, tokens, codes_first_layer)

        # Log audio
        encodec = EncodecPip(device='cpu')
        output_audio = encodec.decode(output_codes.cpu())
        target_audio = encodec.decode(codes.cpu())
        self.logger.experiment.add_audio(
            'val/output_audio', output_audio, sample_rate=encodec.sampling_rate
        )
        self.logger.experiment.add_audio(
            'val/target_audio', target_audio, sample_rate=encodec.sampling_rate
        )

    def forward(
        self,
        tokens: torch.Tensor,
        codes: torch.Tensor,
        codes_lens: torch.Tensor,
        tokens_lens: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Token sequences (batch_size, tokens_len).
            codes: Audio codes (batch_size, codes_len, quantization_layers).
            codes_lens: Lengths of input codes (batch_size).
            tokens_lens: Lengths of input tokens (batch_size).
            layer: Layer to train on.

        Returns:
            logits: Logits (batch_size, tokens_len, num_audio_tokens).
        """
        # pylint: disable=arguments-differ
        # Prepare tokens
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Prepare codes
        codes = self.audio_position_emb(codes)

        # Prepare mask
        codes_pad_mask = F.pad(
            build_pad_mask(codes_lens, self.device),
            (int(tokens_lens.max().item()), 0),
            value=False,
        )

        # Concatenate tokens and codes
        transformer_input = torch.cat([tokens, codes], dim=1)

        # Forward pass
        transformer_output, _ = self.transformer(
            transformer_input,
            padding_mask=codes_pad_mask,
            embedding=self.stage_embs[layer - 1].weight,
        )
        transformer_output = transformer_output[
            :, int(tokens_lens.max().item()) + int(codes_lens.max().item()) :
        ]

        # Project to output
        logits = self.proj_layers[layer - 1](transformer_output)
        logits = rearrange(logits, 'b t q -> b q t')

        return logits

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
            prompt_codes: Audio codes (quantization_layers, prompt_codes_len).
            target_tokens: Target token sequences (target_tokens_len).
            target_codes_first_layer: Target audio codes (target_codes_len).

        Returns:
            output_codes: Output audio codes (output_len, quantization_layers).
        """
        # Prepare prompts
        output_codes = rearrange(target_codes_first_layer, 'q -> 1 q')
        num_quantizers, prompt_len = prompt_codes.shape
        emb_prompt_codes = self.codes_embs[0](prompt_codes[0])
        for j in range(1, num_quantizers):
            emb_prompt_codes += self.codes_embs[j](prompt_codes[j])

        # Prepare tokens
        tokens = rearrange(torch.cat([prompt_tokens, target_tokens], dim=0), 't -> 1 t')
        _, tokens_len = tokens.shape
        tokens = self.tokens_emb(tokens)
        tokens = self.tokens_position_emb(tokens)

        # Decoding loop
        emb_output_codes = self.codes_embs[0](output_codes[0])
        for n_layer in range(1, num_quantizers):
            # Prepare codes
            codes = rearrange(
                torch.cat([emb_prompt_codes, emb_output_codes], dim=0), 't q -> 1 t q'
            )
            codes = self.audio_position_emb(codes)

            # Transformer
            transformer_input = torch.cat([tokens, codes], dim=1)
            transformer_output, _ = self.transformer(
                transformer_input, embedding=self.stage_embs[n_layer - 1].weight
            )

            # Project to output
            logits = self.proj_layers[n_layer - 1](transformer_output[:, tokens_len + prompt_len :])

            # Sampling
            sampled_tokens = Categorical(logits=logits / self.config.temperature).sample()
            emb_output_codes += self.codes_embs[n_layer](sampled_tokens[0])

            # Update output codes
            output_codes = torch.cat([output_codes, sampled_tokens], dim=0)

        return output_codes

    def _prepare_audio_codes(
        self, codes: torch.Tensor, nar_stage: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare prompt audio.

        Args:
            codes: Audio codes (batch_size, quantization_layers, codes_len).

        Returns:
            y_emb: Prompt audio embeddings (batch_size, codes_len, d_model).
            prefix_len: Length of the prompt audio.
        """
        # Cut 3 seconds of audio or 1/3 of the audio
        _, quantization_layers, codes_len = codes.shape
        prefix_len = min(codes_len // 3, 3 * self.config.quantization_factor)
        prompts_codes: torch.Tensor = self.codes_embs[0](codes[:, 0, :prefix_len])
        emb_codes: torch.Tensor = self.codes_embs[0](codes[:, 0, prefix_len:])
        for j in range(1, quantization_layers):
            prompts_codes += self.codes_embs[j](codes[:, j, :prefix_len])
            if j < nar_stage:
                emb_codes += self.codes_embs[j](codes[:, j, prefix_len:])
        y_emb = torch.cat((prompts_codes, emb_codes), dim=1)

        prefix_len_tensor = repeat(
            torch.tensor(prefix_len, dtype=torch.int32), '-> b', b=codes.shape[0]
        )
        return y_emb, prefix_len_tensor

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
