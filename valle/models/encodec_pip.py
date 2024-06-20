import torch
from encodec import EncodecModel


class EncodecPip:
    def __init__(self):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)

    @torch.inference_mode()
    def encode(self, audio):
        encoded_frames = self.model.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        return codes

    def decode(self, codes):
        return self.model.decode(codes)
