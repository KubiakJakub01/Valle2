import torch
from einops import rearrange
from encodec import EncodecModel


class EncodecPip:
    """Encodec model for audio coding and decoding.

    Attributes:
        model: Encodec model
    """

    def __init__(self):
        """Initialize Encodec model."""
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio into codes.

        Args:
            audio: 1D audio tensor of shape [T]

        Returns:
            codes: Tensor of shape [n_q, T]
        """
        assert audio.dim() == 1, f'Expected 1D audio tensor, got {audio.dim()}D'
        audio = rearrange(audio, 't -> 1 1 t')
        encoded_frames = self.model.encode(audio)
        codes = rearrange(
            torch.cat([encoded[0] for encoded in encoded_frames], dim=-1), '1 q t -> q t'
        )
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes into audio.

        Args:
            codes: Tensor of shape [n_q, T]

        Returns:
            audio: 1D audio tensor of shape [T]
        """
        assert codes.dim() == 2, f'Expected 2D codes tensor, got {codes.dim()}D'
        codes = rearrange(codes, 'q t -> 1 q t')
        codes = rearrange(self.model.decode([(codes, None)]), '1 1 t -> t')
        return codes

    def encode_decode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode and decode audio.

        Args:
            audio: 1D audio tensor of shape [T]

        Returns:
            audio: 1D audio tensor of shape [T]
        """
        codes = self.encode(audio)
        audio = self.decode(codes)
        return audio
