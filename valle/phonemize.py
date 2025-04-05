from phonemizer import phonemize
from phonemizer.separator import Separator

# Define punctuation to preserve explicitly
PUNCTUATION = '.,?!'


def phonemize_text(text: str, language='en-us', backend='espeak') -> list[str]:
    """
    Phonemizes a text string using the phonemizer library.

    Args:
        text (str): The text to phonemize.
        language (str): The language to use for phonemization.

    Returns:
        A list of phonemes for the input text.
    """
    if len(text) == 0:
        return ['-']
    phones = phonemize(
        text,
        language=language,
        backend=backend,
        separator=Separator(phone=' ', word=' - ', syllable='|'),
        punctuation_marks=PUNCTUATION,
        strip=True,
        preserve_punctuation=True,
        njobs=4,
    ).split(' ')
    print(phones)
    return postprocess_phonemes(phones)


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
