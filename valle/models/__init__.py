from .valle_ar import ValleAR
from .valle_nar import ValleNAR

MODEL_DICT = {
    'ValleAR': ValleAR,
    'ValleNAR': ValleNAR,
}


def get_model_class(model_name: str):
    return MODEL_DICT[model_name]


__all__ = ['ValleAR', 'ValleNAR']
