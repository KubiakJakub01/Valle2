import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch import loggers, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import ConfigValle
from ..data import get_dataloaders
from ..models import get_model_class
from ..utils import log_info


def train(hparams_fp: Path):
    config: ConfigValle = ConfigValle.from_json(hparams_fp)
    seed_everything(config.seed)
    model = get_model_class(config.model_name)(config)
    log_info(f'Training model {config.model_name} with config: {config}')

    # Load data
    train_dataloader, valid_dataloader = get_dataloaders(config.model_name, config)

    # Logger
    logger = loggers.TensorBoardLogger(config.log_path, name=config.model_name)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.ckpt_path,
        filename='{step}-{val/loss:.2f}',
        save_top_k=-1,
        every_n_train_steps=config.steps_per_ckpt,
    )

    # Trainer
    trainer = L.Trainer(
        max_steps=config.max_steps,
        log_every_n_steps=config.steps_per_log,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.grad_accum,
        logger=logger,
        val_check_interval=config.steps_per_log,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    args = parser.parse_args()

    train(args.config)
