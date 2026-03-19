from tudataset import set_dataset_main
import os

dataset_name = os.environ.get("DATASET", None)
if dataset_name is None:
    raise Exception("Dataset not defined")

set_dataset_main(dataset_name)

