import pytest
import torch

from valle.models.encodec_pip import EncodecPip


@pytest.mark.parametrize(
    'audio, expected_shapes',
    [
        (torch.randn(16000), (8, 50)),
        (torch.randn(32000), (8, 100)),
        (torch.randn(48000), (8, 150)),
    ],
)
def test_encode(audio: torch.Tensor, expected_shapes: tuple[int, int]):
    encodec = EncodecPip()
    codes = encodec.encode(audio)
    assert codes.dim() == 2
    assert codes.shape == expected_shapes
