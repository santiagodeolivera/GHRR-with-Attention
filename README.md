# GHRR-with-Attention

## How to execute

Execute `python GHRR-with-attention/src` with the following environment variables:

* `MODE`: The execution mode of the program (`RANGE` or `TIMED`)
* `PROGRAM_ID`: The ID of the program to be executed
* `START`: The first step to take
* `END`: The last step to take (if `MODE=RANGE`)
* `MINUTES`: How long the program can take at most (if `MODE=TIMED`)
* `PROPORTION`: Proportion between the training set and the whole train-test dataset
* `GRAPH_HD_ROOT`: The path to the GraphHD project, to compare it with GraphHD
* `DATASET`: The ID of the TUDataset dataset
* `DIMENSIONS`: The number of dimensions of the GHRR HVs
* `MATRIX_SIZE`: The size of the matrix in the GHRR HVs
* `MAX_GPU_MEM`: The amount of GPU memory the program can take (it always takes it all)
* `POS_ENC_MODE`: The mode for positional encoding (1: traditional, 2: alternative)
* `BUNDLING_MODE`: The mode for bundling (1: direct sum, 2: sum 1+1 and normalize)
* `TRAIN_INSTANCES`: The number of models to train
* `TRAIN_ITERATIONS`: The number of iterations to apply to each model (if `PROGRAM_ID` is `REFINE_HD` or `REFINE_HD_1`)
* `ROOT_DIR`: The path to the output storage directory
* `TUDATASET_DIR`: The path to the TUDataset storage directory
