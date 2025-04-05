"""
Script to prepare datasets for VALL-E training.

Performs the following steps:
1. Loads metadata based on dataset format (e.g., LJSpeech).
2. Phonemizes transcriptions.
3. Builds a phoneme vocabulary.
4. Extracts audio codes using Encodec.
5. Saves processed metadata, vocabulary, and codes to an output directory.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm

from ..constants import SAMPLING_RATE
from ..models.encodec_pip import EncodecPip
from ..phonemize import phonemize_text
from ..utils import load_audio, log_error, log_info, log_warning


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset for VALL-E training.')
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='Path to the root directory of the dataset.',
    )
    parser.add_argument(
        '--dataset_format',
        type=str,
        default='ljspeech',
        choices=['ljspeech'],
        help='Format of the input dataset.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Path to save the processed data (metadata, vocab, codes).',
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en-us',
        help="Language for phonemization (e.g., 'en-us', 'fr-fr').",
    )
    parser.add_argument(
        '--resample_rate',
        type=int,
        default=SAMPLING_RATE,
        help='Sample rate to resample audio to before encoding.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for Encodec processing (e.g., "cuda", "cpu").',
    )

    return parser.parse_args()


def load_ljspeech_metadata(dataset_dir: Path) -> list[dict]:
    """Loads LJSpeech metadata from metadata.csv."""
    metadata_path = dataset_dir / 'metadata.csv'
    log_info(f'Loading LJSpeech metadata from: {metadata_path}')
    if not metadata_path.is_file():
        log_error(f'metadata.csv not found in {dataset_dir}')
        sys.exit(1)

    metadata = []
    try:
        with open(metadata_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|', quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) != 3:
                    log_warning(f'Skipping malformed row in metadata.csv: {row}')
                    continue
                wav_name, text_normalized, _ = row
                wav_path = dataset_dir / 'wavs' / f'{wav_name}.wav'
                if not wav_path.is_file():
                    log_warning(f'Audio file not found, skipping: {wav_path}')
                    continue
                metadata.append(
                    {
                        '__audio_path': str(wav_path),
                        'id': wav_name,
                        'text': text_normalized,
                    }
                )
        log_info(f'Loaded {len(metadata)} valid entries from LJSpeech metadata.')
        return metadata
    except Exception as e:
        log_error(f'Error reading LJSpeech metadata {metadata_path}: {e}')
        sys.exit(1)


def load_metadata(dataset_dir: Path, dataset_format: str) -> list[dict]:
    """Loads dataset metadata based on the specified format."""
    if dataset_format == 'ljspeech':
        return load_ljspeech_metadata(dataset_dir)
    raise ValueError(f'Unsupported dataset format: {dataset_format}')


def phonemize_metadata(metadata: list[dict], language: str) -> list[dict]:
    """Adds phonemized text to each metadata entry."""
    log_info(f"Starting phonemization for {len(metadata)} entries using language '{language}'...")
    texts_to_phonemize = [entry['text'] for entry in metadata]

    try:
        phonemized_texts = phonemize_text(texts_to_phonemize, language=language)
    except Exception as e:
        log_error(f'Error during phonemization: {e}')
        sys.exit(1)

    if len(phonemized_texts) != len(metadata):
        log_error(
            f'Mismatch between number of texts ({len(metadata)})'
            f'and phonemized results ({len(phonemized_texts)}).'
        )
        sys.exit(1)

    for i, entry in enumerate(metadata):
        entry['phonemes'] = ' '.join(phonemized_texts[i])

    log_info('Phonemization complete.')
    return metadata


def build_and_save_vocabulary(metadata: list[dict], output_file: Path):
    """Builds and saves the phoneme vocabulary from phonemized metadata."""
    log_info('Building phoneme vocabulary...')
    unique_phonemes = set()
    for entry in metadata:
        phonemes_list = entry.get('phonemes', '').split()
        if phonemes_list:
            unique_phonemes.update(phonemes_list)

    log_info(f'Found {len(unique_phonemes)} unique phonemes.')
    if not unique_phonemes:
        log_warning('No phonemes found to build vocabulary.')
        return

    # Sort for consistency
    sorted_phonemes = sorted(list(unique_phonemes))

    vocab_dict = {str(i): phoneme for i, phoneme in enumerate(sorted_phonemes, start=1)}

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
        log_info(f'Vocabulary saved successfully to {output_file}')
    except Exception as e:
        log_error(f'Error writing vocabulary JSON file {output_file}: {e}')
        sys.exit(1)


def extract_and_save_codes(
    metadata: list[dict],
    output_dir: Path,
    resample_rate: int,
    device: str,
):
    """Extracts Encodec codes for audio files and saves them."""
    codes_dir = output_dir / 'codes'
    codes_dir.mkdir(parents=True, exist_ok=True)
    log_info(f'Initializing Encodec model on device: {device}')
    encodec_pip = EncodecPip(device)

    log_info(f'Starting audio code extraction for {len(metadata)} files...')

    crash_count = 0
    for entry in tqdm(metadata, desc='Extracting Codes', unit='file'):
        audio_path = Path(entry.pop('__audio_path'))
        output_code_path = codes_dir / f'{audio_path.stem}.pt'

        if output_code_path.exists():
            continue

        try:
            audio, _ = load_audio(audio_path, resample_rate)
            audio = rearrange(audio, '1 t -> t').to(device)
            codes = encodec_pip.encode(audio)
            torch.save(codes, output_code_path)

        except Exception as e:
            log_error(f'Failed to process {audio_path}: {e}')
            crash_count += 1

    log_info(f'Audio code extraction complete. Codes saved in {codes_dir}')
    log_info(f'Processed {len(metadata) - crash_count} files out of {len(metadata)}')


def save_processed_metadata(metadata: list[dict], output_file: Path):
    """Saves the processed metadata (including phonemes) to a TSV file."""
    log_info(f'Saving processed metadata to {output_file}')
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            if not metadata:
                log_warning('No metadata to save.')
                return
            header = list(metadata[0].keys())
            writer = csv.DictWriter(f, fieldnames=header, delimiter='	')
            writer.writeheader()
            writer.writerows(metadata)
        log_info('Processed metadata saved successfully.')
    except Exception as e:
        log_error(f'Error writing processed metadata file {output_file}: {e}')
        sys.exit(1)


def main(
    dataset_dir: Path,
    dataset_format: str,
    output_dir: Path,
    language: str,
    resample_rate: int,
    device: str,
):
    # 1. Load metadata
    metadata = load_metadata(dataset_dir, dataset_format)
    if not metadata:
        log_error('No metadata loaded. Exiting.')
        sys.exit(1)

    # 2. Phonemize text in metadata
    metadata = phonemize_metadata(metadata, language)

    # 3. Build and save vocabulary
    vocab_path = output_dir / 'vocab.json'
    build_and_save_vocabulary(metadata, vocab_path)

    # 4. Extract and save audio codes
    extract_and_save_codes(metadata, output_dir, resample_rate, device)

    # 5. Save processed metadata (including paths and phonemes)
    processed_metadata_path = output_dir / 'metadata.tsv'
    save_processed_metadata(metadata, processed_metadata_path)

    log_info('Dataset preparation finished successfully.')


if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(**vars(args))
