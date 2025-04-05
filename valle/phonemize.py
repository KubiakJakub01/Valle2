import re
import string

from phonemizer import phonemize
from phonemizer.separator import Separator

# Define punctuation to preserve explicitly
PUNCTUATION = ".,?!'"
PUNCTUATION_TO_REMOVE = set(string.punctuation) - set(PUNCTUATION)
ESCAPED_PUNCTUATION_TO_REMOVE = [re.escape(char) for char in sorted(list(PUNCTUATION_TO_REMOVE))]
REMOVE_PATTERN_STR = '[' + ''.join(ESCAPED_PUNCTUATION_TO_REMOVE) + ']'
REMOVE_PATTERN = re.compile(REMOVE_PATTERN_STR)


def phonemize_text(texts: list[str], language='en-us', backend='espeak') -> list[list[str]]:
    """
    Phonemizes a text string using the phonemizer library.

    Args:
        texts: List of texts to phonemize.
        language: The language to use for phonemization.

    Returns:
        A list of phonemes for the input text.
    """
    texts = [REMOVE_PATTERN.sub('', text) for text in texts]
    phones = phonemize(
        texts,
        language=language,
        backend=backend,
        separator=Separator(phone=' ', word=' - ', syllable='|'),
        punctuation_marks=PUNCTUATION,
        strip=True,
        preserve_punctuation=True,
        njobs=1,
    )
    return [postprocess_phonemes(phone.split(' ')) for phone in phones]


def postprocess_phonemes(phonemes: list[str]) -> list[str]:
    """
    Postprocesses a list of phonemes to separate trailing punctuation.

    Args:
        phonemes: A list of strings representing phonemes, potentially
                  with trailing punctuation attached.

    Returns:
        A new list of strings where trailing punctuation marks
        are separated into their own elements.
    """
    processed_phonemes = []
    previous_phoneme = ''

    for phoneme in phonemes:
        if len(phoneme) > 1 and phoneme[-1] in PUNCTUATION:
            processed_phonemes.append(phoneme[:-1])
            processed_phonemes.append(phoneme[-1])
            previous_phoneme = phoneme[-1]
        elif phoneme == '-' and previous_phoneme in PUNCTUATION:
            continue
        else:
            processed_phonemes.append(phoneme)
            previous_phoneme = phoneme
    if processed_phonemes[-1] not in PUNCTUATION:
        processed_phonemes.append('.')
    return processed_phonemes
