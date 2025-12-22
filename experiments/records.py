import exp_encode_hvs
import exp_register_labels
from experiments import Experiment

def get_experiments() -> dict[str, Experiment]:
    return { \
        "encode-hvs": exp_encode_hvs.experiment, \
        "register-labels": exp_register_labels.experiment \
    }
