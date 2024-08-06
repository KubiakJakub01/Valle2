import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch import loggers, seed_everything

from .config import ConfigValle
from .data import get_dataloaders
from .models import get_model_class
from .utils import log_info


def train(hparams_fp: Path, model_name: str):
    config = ConfigValle.from_json(hparams_fp)
    seed_everything(config.seed)
    model = get_model_class(model_name)(config)

    # Train model
    log_info('Training model %s with hparams: ', model_name, config)

    # Load data
    train_dataloader = get_dataloaders(model_name, config, 'train')

    # Logger
    logger = loggers.TensorBoardLogger(config.log_path, name=model_name)

    # Train model
    trainer = L.Trainer(
        max_steps=config.max_steps,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.grad_accum,
        logger=logger,
    )
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument('-m', '--model', type=str, choices=['ValleAR', 'ValleNAR'], required=True)
    args = parser.parse_args()

    train(args.hparams, args.model)
