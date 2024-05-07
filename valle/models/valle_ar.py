import torch.nn as nn

from ..hparams import ValleHparams
from .modules import SinusodialEmbedding


class ValleAR(nn.Module):
    def __init__(self, hparams: ValleHparams):
        super().__init__()
        self.hparams = hparams

        # Embeddings
        self.text_emb = nn.Embedding(hparams.vocab_size, hparams.d_model)
        self.audio_emb = nn.Embedding(hparams.num_audio_tokens, hparams.d_model)
        self.text_position_emb = SinusodialEmbedding(hparams.d_model)
        self.audio_position_emb = SinusodialEmbedding(hparams.d_model)

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hparams.d_model,
                nhead=hparams.n_head,
                dim_feedforward=hparams.dim_feedforward,
                dropout=hparams.dropout,
                activation=hparams.activation,
            ),
            num_layers=hparams.num_layers,
        )

        # Project to output
        self.proj = nn.Linear(hparams.d_model, hparams.num_audio_tokens + 1, bias=False)

    def forward(self, x):
        return x
