import torch

from valle.config import ConfigValle
from valle.models import ValleAR


def test_valle_ar():
    config = ConfigValle(norm='LayerNorm')
    model = ValleAR(config)
    tokens_list = [torch.randint(0, 256, (10,)) for _ in range(4)]
    codes_list = [torch.randint(0, 256, (10,)) for _ in range(4)]
    loss = model(tokens_list, codes_list)

    assert loss is not None
    assert loss.dim() == 0
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.item(), float)
