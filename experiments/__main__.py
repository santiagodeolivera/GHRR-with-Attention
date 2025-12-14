import argparse
from experiments import Experiment, get_experiments

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
    experiment: Experiment | None = get_experiments().get(experiment_id, None)
    if 

if __name__ == "__main__":
    main()
