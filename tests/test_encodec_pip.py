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


@pytest.mark.parametrize(
    'audios, expected_shapes',
    [
        (torch.randn(4, 16000), (4, 8, 50)),
        (torch.randn(4, 32000), (4, 8, 100)),
        (torch.randn(4, 48000), (4, 8, 150)),
    ],
)
def test_batch_encode(audios: torch.Tensor, expected_shapes: tuple[int, int, int]):
    encodec = EncodecPip()
    codes = encodec.batch_encode(audios)
    assert codes.dim() == 3
    assert codes.shape == expected_shapes
