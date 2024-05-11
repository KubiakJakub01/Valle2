import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from ..hparams import ValleHparams
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

    def forward(self, tokens_list: list[torch.Tensor], codes_list: list[torch.Tensor]):
        assert len(tokens_list) == len(codes_list), 'Batch size mismatch.'

        # Prepare tokens
        x = pad_sequence(tokens_list, batch_first=True)
        x = self.tokens_emb(x)  # (b t c)
        x = self.tokens_position_emb(x)
