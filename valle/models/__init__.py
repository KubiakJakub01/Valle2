from .encodec_pip import EncodecPip
from .valle_ar import ValleAR
from .valle_nar import ValleNAR

MODEL_DICT = {
    'encodec_pip': EncodecPip,
    'valle_ar': ValleAR,
    'valle_nar': ValleNAR,
}


def get_model_class(model_name: str):
    return MODEL_DICT[model_name]


__all__ = ['encodec_pip', 'valle_ar', 'valle_nar']
