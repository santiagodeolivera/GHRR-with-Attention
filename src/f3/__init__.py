from typing import Callable
import subprocess
import json
from pathlib import Path
from utils import approximation as get_approximation
import shutil
from tudataset import get_dataset_main
from fn_context import FnContext
from get_args import get_arg

def execute_graphhd(ctx: FnContext) -> None:
    root = ctx.fs
    
    counter = 0
    
    output_root = root.root / "comparisons"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)
    
    for instance_id in range(10):
        root.config.dist_file = f"instances/{instance_id}/dist_file.json"
        graphhd_root = get_arg("GRAPH_HD_ROOT", "Path")
        subprocess.run(["python", "main.py", f"--distr_file={root.train_and_test_sets_distribution}", f"--dataset={get_dataset_main().name}"], cwd=graphhd_root)
        
        root.config.result_file = f"instances/{instance_id}/result_file.json"
        with open(root.result_file, "r") as ghrr_result_file:
            # {"ids": int[], "expected": int[], "result": int[]}
            ghrr_result = json.load(ghrr_result_file)
        ghrr_predictions = ghrr_result["result"]
        
        for graphhd_result_path in (graphhd_root / "individual_results").iterdir():
            with open(graphhd_result_path, "r") as graphhd_result_file:
                # {"dataset": str, "enc_name": str, "metric": str, "iteration": int, \
                # "ids": int[], "expected": int[], "result": int[]}
                graphhd_result = json.load(graphhd_result_file)
            graphhd_result["expected"] = [(1 if x == 1 else 0) for x in graphhd_result["expected"]]
            graphhd_result["result"] = [(1 if x == 1 else 0) for x in graphhd_result["result"]]
            
            graphhd_predictions = graphhd_result["result"]
            coincidences = tuple(graphhd_predictions[i] == ghrr_predictions[i] \
                    for i in range(len(ghrr_predictions)))
            total_coincidences = sum(1 for x in coincidences if x)
            
            if ghrr_result["ids"] != graphhd_result["ids"]:
                print("GHRR ids:", ghrr_result["ids"])
                print("GraphHD ids:", graphhd_result["ids"])
                raise Exception()

            if ghrr_result["expected"] != graphhd_result["expected"]:
                print("GHRR expected:", ghrr_result["expected"])
                print("GraphHD expected:", graphhd_result["expected"])
                raise Exception()
            
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
                "total_coincidences": total_coincidences,
                "proportion_coincidences": total_coincidences / len(coincidences)
            }
            
            output_path = output_root / f"{counter}.json"
            print(f"Printing to {output_path}")
            
            with open(output_path, "w") as output_file:
                json.dump(result_data, output_file)
            counter += 1

def find_closeness_approximation(ctx: FnContext) -> None:
    root = ctx.fs
    
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
        
# Compare with GraphHD
def get_action(action_id: int) -> tuple[str, Callable[[FnContext], None]] | None:
    if action_id < 0:
        raise ValueError("Out of bounds")
    elif action_id == 0:
        return ("Calculate comparisons with GraphHD", execute_graphhd)
    elif action_id == 1:
        return ("Aggregate comparisons with GraphHD", find_closeness_approximation)
    return None

