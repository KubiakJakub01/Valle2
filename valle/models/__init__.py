from .encodec_pip import EncodecPip
from .valle_ar import ValleAR
from .valle_nar import ValleNAR

MODEL_DICT = {
    'EncodecPip': EncodecPip,
    'ValleAR': ValleAR,
    'ValleNAR': ValleNAR,
}


def get_model_class(model_name: str):
    return MODEL_DICT[model_name]


__all__ = ['EncodecPip', 'ValleAR', 'ValleNAR']
