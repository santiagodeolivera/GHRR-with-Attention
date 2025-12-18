import exp_1
from experiments import Experiment

def get_experiments() -> dict[str, Experiment]:
    return {                   \
        "1": exp_1.experiment  \
    }
