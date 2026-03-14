from typing import Callable, Any
from fs_organization import FsOrganizer
import subprocess
import json
import os
from pathlib import Path
from utils import approximation as get_approximation

def get_parameter(name: str) -> str:
    result = os.environ.get(name, None)
    if result is None:
        raise Exception(f"Parameter not found: {repr(name)}")
    
    return result

def execute_graphhd(root: FsOrganizer) -> None:
    counter = 0
    for instance_id in range(10):
        root.config.dist_file = f"instances/{instance_id}/dist_file.json"
        graphhd_root = Path(get_parameter("GRAPH_HD_ROOT"))
        subprocess.run(["python", "main.py", f"--distr_file={root.train_and_test_sets_distribution}"], cwd=graphhd_root)
        
        root.config.result_file = f"instances/{instance_id}/result_file.json"
        with open(root.result_file, "r") as ghrr_result_file:
            # {"ids": int[], "expected": int[], "result": int[]}
            ghrr_result = json.load(ghrr_result_file)
        ghrr_predictions = ghrr_result["result"]
        
        for graphhd_result_path in (graphhd_root / "individual_results").iterdir():
            with open(graphhd_result_path, "r") as graphhd_result_file:
                # {"dataset": str, "enc_name": str, "metric": str, "iteration": int, "predictions": int[]}
                graphhd_result = json.load(graphhd_result_file)
            graphhd_predictions = tuple((1 if x == 1 else 0) for x in graphhd_result["predictions"])
            coincidences = tuple(graphhd_predictions[i] == ghrr_predictions[i] \
                    for i in range(len(ghrr_predictions)))
            result_data = {
                "ghrr_id": instance_id,
                "graphhd_id": {
                    "enc_name": graphhd_result["enc_name"],
                    "metric": graphhd_result["metric"],
                    "iteration": graphhd_result["iteration"]
                },
                "ids": ghrr_result["ids"],
                "expected_labels": ghrr_result["expected"],
                "ghrr_result": ghrr_predictions,
                "graphhd_result": graphhd_predictions,
                "coincidences": coincidences,
                "total_coincidences": sum(1 for x in coincidences if x),
                "proportion_coincidences": sum(1 for x in coincidences if x) / len(coincidences)
            }
            
            output_path = root.root / f"comparisons/{counter}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Printing to {output_path}")
            
            with open(output_path, "w") as output_file:
                json.dump(result_data, output_file)
            counter += 1

def find_closeness_approximation(root: FsOrganizer) -> None:
    def get_proportion_coincidences(path: Path) -> float:
        with open(path, "r") as file:
            data = json.load(file)
        result = data["proportion_coincidences"]
        if type(result) != float: raise Exception(f"Expected a float, got {repr(result)}")
        return result
    
    all_coincidences = tuple(get_proportion_coincidences(path) for path in (root.root / "comparisons").iterdir())
    approximation = get_approximation(all_coincidences)
    with open(root.root / "comparisons_approximation.json", "w") as file:
        json.dump(approximation, file)
        

def get_action(action_id: int) -> Callable[[FsOrganizer], None]:
    if action_id < 0:
        raise ValueError("Out of bounds")
    elif action_id == 0:
        return execute_graphhd
    elif action_id == 1:
        return find_closeness_approximation
    else:
        raise ValueError("Out of bounds")

