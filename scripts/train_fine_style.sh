#!/bin/bash

GPU_COUNT=1
MAX_NUM=27

for ((folder_number = 0; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        CUDA_VISIBLE_DEVICES=`expr $gpu_id + 0`
        STYLE_SET=$(printf "%02d" $current_folder_number)

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python fine_style_training.py --style=$STYLE_SET"

        eval $COMMAND &
        sleep 2
    done
    wait
done

