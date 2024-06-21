import torch
from einops import rearrange
from encodec import EncodecModel


class EncodecPip:
    def __init__(self):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio into codes.

        Args:
            audio: 1D audio tensor of shape [T]

        Returns:
            codes: Tensor of shape [B, n_q, T]
        """
        assert audio.dim() == 1, f'Expected 1D audio tensor, got {audio.dim()}D'
        audio = rearrange(audio, 't -> 1 1 t')
        encoded_frames = self.model.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes into audio.

        Args:
            codes: Tensor of shape [B, n_q, T]

        Returns:
            audio: 1D audio tensor of shape [T]
        """
        codes = rearrange(self.model.decode([(codes, None)]), '1 1 t -> t')
        return codes
