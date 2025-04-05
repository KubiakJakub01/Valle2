import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .collate import get_collate
from .config import ConfigValle
from .utils import load_tsv, log_error, log_info, log_warning


class ValleDataset(Dataset):
    """Dataset for loading preprocessed VALL-E data."""

    def __init__(self, metadata_list: list[dict], config: ConfigValle):
        """
        Initializes the dataset.

        Args:
            metadata_list: A list of dictionaries, where each dict represents an item
                           from the processed metadata.tsv file.
            config: Configuration object containing paths and settings.
        """
        super().__init__()
        self.metadata = metadata_list
        self.config = config
        self.phoneme_to_id = self._load_vocabulary(self.config.vocab_path)

        log_info(f'Initialized ValleDataset with {len(self.metadata)} items.')
        log_info(f'Vocabulary size: {len(self.phoneme_to_id)}')

    def _load_vocabulary(self, vocab_path: Path) -> dict[str, int]:
        """Loads the phoneme vocabulary and creates a phoneme-to-ID mapping."""
        log_info(f'Loading vocabulary from: {vocab_path}')
        try:
            with open(vocab_path, encoding='utf-8') as f:
                vocab_data = json.load(f)
            phoneme_to_id = {phoneme: int(id_str) for id_str, phoneme in vocab_data.items()}
            return phoneme_to_id
        except Exception as e:
            log_error(f'An unexpected error occurred while loading vocabulary: {e}')
            raise

    def _tokenize_phonemes(self, phoneme_string: str) -> torch.Tensor:
        """Converts a space-separated phoneme string into a tensor of integer IDs."""
        tokens = []
        for phoneme in phoneme_string.strip().split():
            token_id = self.phoneme_to_id.get(phoneme)
            if token_id is None:
                log_warning(f"Phoneme '{phoneme}' not found in vocabulary. Skipping.")
                continue
            tokens.append(token_id)
        return torch.LongTensor(tokens)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Loads and returns a single data item.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            A dictionary containing:
                'codes': Pre-extracted audio codes (torch.Tensor [N_Q, T_codes]).
                'tokens': Phoneme IDs (torch.Tensor [T_tokens]).
        """
        item = self.metadata[idx]
        item_id = item.get('id')
        phonemes = item.get('phonemes')

        if item_id is None or phonemes is None:
            log_error(f'Metadata item at index {idx} is missing "id" or "phonemes". Item: {item}')
            raise ValueError(f'Invalid metadata item at index {idx}')

        code_path = self.config.codes_dir / f'{item_id}.pt'

        try:
            codes = torch.load(code_path, map_location='cpu')
            if not isinstance(codes, torch.Tensor):
                raise TypeError(f'Loaded codes file is not a tensor: {code_path}')

            tokens = self._tokenize_phonemes(phonemes)

            return {'codes': codes, 'tokens': tokens}

        except FileNotFoundError:
            log_error(f'Code file not found for item {item_id}: {code_path}')
            raise
        except Exception as e:
            log_error(f'Error loading or processing item {item_id} (idx {idx}): {e}')
            raise


def get_dataloaders(model_name: str, config: ConfigValle) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders from the prepared dataset directory.

    Args:
        model_name: Name of the model (used potentially by collate function).
        config: Configuration object with paths and hyperparameters.

    Returns:
        A tuple containing the training DataLoader and validation DataLoader.
    """
    log_info(f'Loading metadata from: {config.metadata_path}')
    full_metadata = list(
        load_tsv(config.metadata_path, columns={'id': 'id', 'phonemes': 'phonemes'})
    )

    if not full_metadata:
        log_error('Metadata file is empty. Cannot create dataloaders.')
        raise ValueError('No data found in metadata file.')

    # Shuffle and split metadata
    random.seed(config.seed)  # Use seed for reproducible splits
    random.shuffle(full_metadata)
    split_idx = int(len(full_metadata) * 0.9)  # 90/10 split
    train_metadata = full_metadata[:split_idx]
    valid_metadata = full_metadata[split_idx:]

    log_info(f'Train set size: {len(train_metadata)}')
    log_info(f'Validation set size: {len(valid_metadata)}')

    # Create Datasets
    train_dataset = ValleDataset(train_metadata, config)
    valid_dataset = ValleDataset(valid_metadata, config)

    # Create DataLoaders
    collate_fn = get_collate(model_name)(config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader
