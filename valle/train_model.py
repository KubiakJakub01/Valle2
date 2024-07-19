import argparse
from pathlib import Path
from typing import get_args

import lightning as L
from lightning.pytorch import seed_everything

from .data import get_dataloaders
from .hparams import ValleHparams
from .models import MODEL_DICT, get_model_class
from .utils import log_info


def train(hparams_fp: Path, model_name: str):
    hparams = ValleHparams.from_json(hparams_fp)
    seed_everything(hparams.seed)
    model = get_model_class(model_name)(hparams)

    # Train model
    log_info('Training model %s with hparams: ', model_name, hparams)

    # Load data
    train_dataloader = get_dataloaders(hparams, 'train')
    val_dataloader = get_dataloaders(hparams, 'val')

    # Train model
    trainer = L.Trainer(max_steps=hparams.max_steps, log_every_n_steps=hparams.log_every_n_steps)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=hparams.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=Path, required=True)
    parser.add_argument('--model', type=str, choices=get_args(MODEL_DICT), required=True)
    args = parser.parse_args()

    train(args.hparams, args.model)
