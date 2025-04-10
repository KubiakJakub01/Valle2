import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

from ..config import ConfigValle
from ..models import EncodecPip, ValleAR, ValleNAR, load_model_for_inference
from ..phonemize import phonemize_text
from ..utils import load_audio, load_tsv, log_error, log_info, tokenize_phonemes


@dataclass
class SynthesizeItem:
    """Item for synthesis."""

    item_id: str
    target_text: str
    prompt_audio_path: Path
    prompt_text: str
    target_tokens: torch.Tensor
    prompt_tokens: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Synthesize speech using VALL-E AR and NAR models.'
    )
    parser.add_argument(
        '--config_ar', type=Path, required=True, help='Path to the ValleAR configuration file.'
    )
    parser.add_argument(
        '--config_nar', type=Path, required=True, help='Path to the ValleNAR configuration file.'
    )
    parser.add_argument(
        '--input_path', type=Path, help='Path to the TSV file containing the input items.'
    )
    parser.add_argument(
        '--output_path', type=Path, required=True, help='Path to save the synthesized audio (.wav).'
    )
    return parser.parse_args()


def load_synthesize_items(
    input_path: Path,
    config: ConfigValle,
) -> list[SynthesizeItem]:
    """Load synthesize items from the given path and text."""
    items = load_tsv(input_path)
    target_phonemes = phonemize_text([item['target_text'] for item in items], config.language)
    prompt_phonemes = phonemize_text([item['prompt_text'] for item in items], config.language)
    target_tokens = [
        tokenize_phonemes(phonemes, config.phoneme_to_id) for phonemes in target_phonemes
    ]
    prompt_tokens = [
        tokenize_phonemes(phonemes, config.phoneme_to_id) for phonemes in prompt_phonemes
    ]

    return [
        SynthesizeItem(
            item_id=item['item_id'],
            target_text=item['target_text'],
            prompt_audio_path=Path(item['prompt_audio_path']),
            prompt_text=item['prompt_text'],
            target_tokens=target_token,
            prompt_tokens=prompt_token,
        )
        for item, target_token, prompt_token in zip(
            items, target_tokens, prompt_tokens, strict=False
        )
    ]


@torch.inference_mode()
def synthesize_item(
    item: SynthesizeItem,
    valle_ar: ValleAR,
    valle_nar: ValleNAR,
    encodec_pip: EncodecPip,
    config: ConfigValle,
    device: torch.device,
) -> torch.Tensor:
    """Synthesize an item."""
    # Load prompt audio
    audio, _ = load_audio(item.prompt_audio_path, config.sampling_rate)
    audio = rearrange(audio, '1 t -> t').to(device)
    codes = encodec_pip.encode(audio)

    prompt_tokens = item.prompt_tokens.to(device)
    target_tokens = item.target_tokens.to(device)

    # Synthesize
    valle_ar_output = valle_ar.generate(
        prompt_tokens=prompt_tokens,
        prompt_codes=codes,
        target_tokens=target_tokens,
    )
    valle_nar_output = valle_nar.generate(
        prompt_tokens=prompt_tokens,
        prompt_codes=codes,
        target_tokens=target_tokens,
        target_codes_first_layer=valle_ar_output,
    )
    output_audio = rearrange(encodec_pip.decode(valle_nar_output), 't -> 1 t')

    return output_audio.cpu()


def main(
    config_ar_fp: Path,
    config_nar_fp: Path,
    input_path: Path,
    output_path: Path,
):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info(f'Using device: {device}')
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models
    try:
        valle_ar_config = ConfigValle.from_json(config_ar_fp)
        valle_nar_config = ConfigValle.from_json(config_nar_fp)
        valle_ar: ValleAR = load_model_for_inference(valle_ar_config, ValleAR, device)
        valle_nar: ValleNAR = load_model_for_inference(valle_nar_config, ValleNAR, device)
        encodec_pip = EncodecPip(device=device)
        log_info('Models loaded.')
    except Exception as e:
        log_error(f'Failed to load models: {e}')
        return

    # Preprocess inputs
    log_info('Preprocessing inputs...')
    try:
        synthesize_items = load_synthesize_items(input_path, valle_ar_config)
        log_info(f'Loaded {len(synthesize_items)} synthesize items.')
    except Exception as e:
        log_error(f'Error preprocessing inputs: {e}')
        return

    # Synthesize
    log_info('Synthesizing...')
    for item in synthesize_items:
        log_info(f'Synthesizing item: {item.item_id}')
        try:
            output_audio = synthesize_item(
                item, valle_ar, valle_nar, encodec_pip, valle_ar_config, device
            )
            torchaudio.save(
                output_path / f'{item.item_id}.wav', output_audio, valle_ar_config.sampling_rate
            )
        except Exception as e:
            log_error(f'Error synthesizing item {item.item_id}: {e}')
            continue


if __name__ == '__main__':
    args = parse_args()
    main(
        args.config_ar,
        args.config_nar,
        args.input_path,
        args.output_path,
    )
