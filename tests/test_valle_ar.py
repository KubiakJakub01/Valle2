import pytest
import torch

from valle.hparams import ValleHparams
from valle.models.valle_ar import ValleAR


def test_valle_ar():
    hparams = ValleHparams(norm='LayerNorm')
    model = ValleAR(hparams)
    tokens_list = [torch.randint(0, 256, (10,)) for _ in range(4)]
    codes_list = [torch.randint(0, 256, (10,)) for _ in range(4)]
    loss = model(tokens_list, codes_list)

    print(f'Loss: {loss}')

    with pytest.raises(AssertionError):
        model(tokens_list, codes_list[:-1])

    with pytest.raises(AssertionError):
        model(tokens_list[:-1], codes_list)

    with pytest.raises(AssertionError):
        model(tokens_list[:-1], codes_list[:-1])


if __name__ == '__main__':
    test_valle_ar()
