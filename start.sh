#!/bin/bash

# Memory error prevention
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Parameters:"
echo "Start: $START"
echo "End: $END"
echo "Root directory: $ROOT_DIR"
echo "Memory history directory: $MEM_HISTORY_DIR"
echo "Train-test datasets proportion: $PROPORTION"
echo ""

if [[ $START == "" ]]; then
	START=0
fi

exec_path=$(dirname "$0")
exec_path=$exec_path/src

i=$START

while true; do
	echo "Starting action $i"
	export ACTION_ID=$i
	export MEM_HISTORY_OUT=$MEM_HISTORY_DIR/$i.pkl
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

