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


@pytest.mark.parametrize(
    'codes, expected_shape',
    [
        (torch.randint(0, 1024, (8, 50)), (16000,)),
        (torch.randint(0, 1024, (8, 100)), (32000,)),
        (torch.randint(0, 1024, (8, 150)), (48000,)),
    ],
)
def test_decode(codes: torch.Tensor, expected_shape: tuple[int]):
    encodec = EncodecPip()
    audio = encodec.decode(codes)
    assert audio.dim() == 1
    assert audio.shape == expected_shape


@pytest.mark.parametrize(
    'codes, expected_shape',
    [
        (torch.randint(0, 1024, (4, 8, 50)), (4, 16000)),
        (torch.randint(0, 1024, (4, 8, 100)), (4, 32000)),
        (torch.randint(0, 1024, (4, 8, 150)), (4, 48000)),
    ],
)
def test_batch_decode(codes: torch.Tensor, expected_shape: tuple[int]):
    encodec = EncodecPip()
    audios = encodec.batch_decode(codes)
    assert audios.dim() == 2
    assert audios.shape == expected_shape


@pytest.mark.parametrize(
    'audio, expected_shapes',
    [
        (torch.randn(16000), (8, 50)),
        (torch.randn(32000), (8, 100)),
        (torch.randn(48000), (8, 150)),
    ],
)
def test_encode_decode(audio: torch.Tensor, expected_shapes: tuple[int, int]):
    encodec = EncodecPip()
    codes = encodec.encode(audio)
    assert codes.dim() == 2
    assert codes.shape == expected_shapes
    output_audio = encodec.decode(codes)
    assert audio.dim() == 1
    assert output_audio.shape == audio.shape


@pytest.mark.parametrize(
    'audios, expected_shapes',
    [
        (torch.randn(16000), (8, 50)),
        (torch.randn(32000), (8, 100)),
        (torch.randn(48000), (8, 150)),
    ],
)
def test_get_embedding(audios: torch.Tensor, expected_shapes: tuple[int, int]):
    encodec = EncodecPip()
    codes = encodec.get_embedding(audios)
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
def test_batch_get_embedding(audios: torch.Tensor, expected_shapes: tuple[int, int, int]):
    encodec = EncodecPip()
    codes = encodec.batch_get_embedding(audios)
    assert codes.dim() == 3
    assert codes.shape == expected_shapes
