import argparse
from experiments import Experiment
from records import get_experiments
from ghrr_with_attention.utils import not_none
from json import dumps as json_dumps

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='GHRR with attention experiments',
        allow_abbrev=False
    )
    parser.add_argument("experiment_id")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser

def main() -> None:
    args = get_parser().parse_args()

    experiment_id: str = args.experiment_id
    rest: Iterable[str] = args.rest
    
    experiment: Experiment | None = get_experiments().get(experiment_id, None)
    
    if not not_none(experiment):
        v1 = json_dumps(experiment_id)
        print(f"Invalid experiment id: {v1}")
        return
    
    experiment.execute(rest)

if __name__ == "__main__":
    main()
