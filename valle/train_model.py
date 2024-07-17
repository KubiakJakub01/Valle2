import argparse
from pathlib import Path
from typing import get_args

from lightning.pytorch import seed_everything

from .hparams import ValleHparams
from .models import MODEL_DICT, get_model_class
from .utils import log_info


def train(hparams_fp: Path, model_name: str):
    hparams = ValleHparams.from_json(hparams_fp)
    seed_everything(hparams.seed)
    model = get_model_class(model_name)(hparams)

    # Train model
    log_info(f'Training model {model_name} with hparams: {hparams}')
    log_info(f'Model: {model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=Path, required=True)
    parser.add_argument('--model', type=str, choices=get_args(MODEL_DICT), required=True)
    args = parser.parse_args()

    train(args.hparams, args.model)
