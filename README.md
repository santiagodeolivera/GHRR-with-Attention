# GHRR-with-Attention

## Note

In order for the program to be executable, it's divided into several steps, each of which is done via executing the program with a different `ACTION_ID` parameter, which currently goes from 0 to 220 (inclusive).

## Executing a single step

1. Make sure all requirements are installed (check the requirements.txt file).
2. Set up the following environment variables:
    * `ACTION_ID`: The ID of the step to be executed.
    * `ROOT_DIR`: The path to the root directory where the intermediate files will be stored.
    * `MEM_HISTORY_OUT`: (Optional) The path to export the GPU memory history during the execution.
    * `PROPORTION`: The proportion of rows of the original dataset to be used for training (the rest will be for testing). Only mandatory in certain steps, but recommended to be set beforehand.
3. Execute python on the `src` directory.

## Executing a range of steps

It can only be done in Windows or Linux operating systems.

1. Make sure all requirements are installed (check the requirements.txt file).
2. Set up the following environment variables:
    * `START`: The ID of the first step to be executed.
    * `END`: The ID of the last step to be executed.
    * `ROOT_DIR`: The path to the root directory where the intermediate files will be stored.
    * `MEM_HISTORY_DIR`: The path to the directory where the GPU memory history files will be stored. This time it's mandatory.
    * `PROPORTION`: The proportion of rows of the original dataset to be used for training (the rest will be for testing). Only mandatory in certain steps, but recommended to be set beforehand.
3. Execute the corresponding file (`start.cmd` if OS is Windows, `start.sh` if it's Linux).

