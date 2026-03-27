from tudataset import set_dataset_main

from get_args import get_arg

dataset_name = get_arg("DATASET", "str")

set_dataset_main(dataset_name)

