from tudataset import set_dataset
from pathlib import Path

from fs_organization import FsOrganizer
from get_args import get_arg

dataset_name = get_arg("DATASET", "str")
root_dir = get_arg("ROOT_DIR", "Path")

set_dataset(dataset_name, FsOrganizer(root_dir).tudataset)

