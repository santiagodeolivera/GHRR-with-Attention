#!/bin/bash

# Memory error prevention
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Parameters:"
echo "Program: $PROGRAM_ID"
echo "Start: $START"
echo "End: $END"
echo "Root directory: $ROOT_DIR"
echo "Memory history directory: $MEM_HISTORY_DIR"
echo "Train-test datasets proportion: $PROPORTION"
echo ""

if [[ "$PROGRAM_ID" == "" ]]; then
	PROGRAM_ID=1
fi

if [[ "$START" == "" ]]; then
	START=0
fi

exec_path=$(dirname "$0")
exec_path=$exec_path/src

i=$START

while true; do
	export ACTION_ID=$i
	echo "Starting action $ACTION_ID"
	export MEM_HISTORY_OUT=$MEM_HISTORY_DIR/$PROGRAM_ID-$ACTION_ID.pkl
	python $exec_path
	el=$?
	echo "Action $i ended"

	if [[ $i == $END ]]; then
		break
	fi

	if [[ $el != 0 ]]; then
		break
	fi

	i=$((i+1))
done

