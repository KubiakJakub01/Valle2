import pytest

from valle.phonemize import phonemize_text


@pytest.mark.parametrize(
    'text, language, expected_output',
    [
        (
            ['Hello, this world!'],
            'en-us',
            [['h', 'ə', 'l', 'oʊ', ',', 'ð', 'ɪ', 's', '-', 'w', 'ɜː', 'l', 'd', '!']],
        ),
        (
            ['This is a test.', 'No punctuation here'],
            'en-us',
            [
                ['ð', 'ɪ', 's', '-', 'ɪ', 'z', '-', 'ɐ', '-', 't', 'ɛ', 's', 't', '.'],
                [
                    'n',
                    'oʊ',
                    '-',
                    'p',
                    'ʌ',
                    'ŋ',
                    'k',
                    'tʃ',
                    'uː',
                    'eɪ',
                    'ʃ',
                    'ə',
                    'n',
                    '-',
                    'h',
                    'ɪɹ',
                    '.',
                ],
            ],
        ),
        (
            ['', 'Hello, this world!'],
            'en-us',
            [[''], ['h', 'ə', 'l', 'oʊ', ',', 'ð', 'ɪ', 's', '-', 'w', 'ɜː', 'l', 'd', '!']],
        ),
        (
            ['Bonjour le monde?'],
            'fr-fr',
            [['b', 'ɔ̃', 'ʒ', 'u', 'ʁ', '-', 'l', 'ə', '-', 'm', 'ɔ̃', 'd', '?']],
        ),
    ],
)
def test_phonemize_text(text: list[str], language: str, expected_output: list[list[str]]):
    """Tests the phonemize_text function with various inputs."""
    result = phonemize_text(text, language=language)
    assert result == expected_output
