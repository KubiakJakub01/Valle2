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
            codes: Tensor of shape [N_Q, T]
        """
        assert audio.dim() == 1, f'Expected 1D audio tensor, got {audio.dim()}D'
        audio = rearrange(audio, 't -> 1 1 t')
        encoded_frames = self.model.encode(audio)
        codes = rearrange(
            torch.cat([encoded[0] for encoded in encoded_frames], dim=-1), '1 q t -> q t'
        )
        return codes

    @torch.inference_mode()
    def batch_encode(self, audios: torch.Tensor) -> torch.Tensor:
        """Encode batch of audio into codes.

        Args:
            audios: 2D audio tensor of shape [B, T]

        Returns:
            codes: Tensor of shape [B, N_Q, T]
        """
        assert audios.dim() == 2, f'Expected 2D audio tensor, got {audios.dim()}D'
        audios = rearrange(audios, 'b t -> b 1 t')
        encoded_frames = self.model.encode(audios)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes into audio.

        Args:
            codes: Tensor of shape [N_Q, T]

        Returns:
            audio: 1D audio tensor of shape [T]
        """
        assert codes.dim() == 2, f'Expected 2D codes tensor, got {codes.dim()}D'
        codes = rearrange(codes, 'q t -> 1 q t')
        codes = rearrange(self.model.decode([(codes, None)]), '1 1 t -> t')
        return codes

    @torch.inference_mode()
    def batch_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode batch of codes into audio.

        Args:
            codes: Tensor of shape [B, N_Q, T]

        Returns:
            audio: 2D audio tensor of shape [B, T]
        """
        assert codes.dim() == 3, f'Expected 3D codes tensor, got {codes.dim()}D'
        audios = self.model.decode([(codes, None)])
        audios = rearrange(audios, 'b 1 t -> b t')
        return audios

    @torch.inference_mode()
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

    @torch.inference_mode()
    def get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Get embedding of audio.

        Args:
            audio: 1D audio tensor of shape [T]

        Returns:
            embedding: Tensor of shape [C, T]
        """
        assert audio.dim() == 1, f'Expected 1D audio tensor, got {audio.dim()}D'
        audio = rearrange(audio, 't -> 1 1 t')
        emb = rearrange(self.model.encoder(audio), '1 c t -> c t')
        return emb

    @torch.inference_mode()
    def batch_get_embedding(self, audios: torch.Tensor) -> torch.Tensor:
        """Get embedding of batch of audio.

        Args:
            audios: 2D audio tensor of shape [B, T]

        Returns:
            embedding: Tensor of shape [B, C, T]
        """
        assert audios.dim() == 2, f'Expected 2D audio tensor, got {audios.dim()}D'
        audios = rearrange(audios, 'b t -> b 1 t')
        emb = self.model.encoder(audios)
        return emb
