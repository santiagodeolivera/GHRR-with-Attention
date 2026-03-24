#!/bin/bash

# Memory error prevention
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Parameters:"
echo "Program: $PROGRAM_ID"
echo "Start: $START"
echo "End: $END"
echo "Dataset: $DATASET"
echo "Root directory: $ROOT_DIR"
echo "Train-test datasets proportion: $PROPORTION"
echo "Path to GraphHD program: $GRAPH_HD_ROOT"
echo "Dimensions: $DIMENSIONS"
echo "Matrix size: $MATRIX_SIZE"
echo ""

if [[ "$PROGRAM_ID" == "" ]]; then
	PROGRAM_ID=1
fi

if [[ "$START" == "" ]]; then
	START=0
fi

exec_path="$(dirname "$0")"
exec_path="$exec_path/src"

python "$exec_path"

