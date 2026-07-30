#!/bin/bash
set -e

# This script is used to preprocess a batch of scenes from the SceneFun3D dataset. 
# It takes a batch ID as input and processes a specified number of scenes in that batch (10 by default). 
# The script sets up the necessary environment variables, defines paths for the dataset 
# and output directories, and iterates over the scenes in the specified batch to run the preprocessing script for each scene.

BATCH_ID=$1  # User input batch ID, starting from 0
BATCH_SIZE=10 # Number of scenes to process in one batch

if [ -z "$BATCH_ID" ]; then
    echo "Usage: $0 <batch_id>"
    exit 1
fi

SPLIT="train"
SCENE_ROOT="$(pwd)/datasets/scenefun3d/$SPLIT"
SCENE_IDS=($(ls "$SCENE_ROOT"))

total_scenes=${#SCENE_IDS[@]}
total_batches=$(( (total_scenes + BATCH_SIZE - 1) / BATCH_SIZE ))

if [ "$BATCH_ID" -ge "$total_batches" ]; then
    echo "Error: batch_id $BATCH_ID out of range. Total batches: $total_batches"
    exit 1
fi

echo "Total scenes: $total_scenes"
echo "Processing batch $BATCH_ID (Batch size: $BATCH_SIZE)"

DATA_BASE="$(pwd)/datasets/scenefun3d"

# Calculate the start and end indices of this batch.
START=$(( BATCH_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))
if [ "$END" -gt "$total_scenes" ]; then
    END=$total_scenes
fi

# Iterate over the scenes in the current batch
for (( i=START; i<END; i++ )); do
    SCENE_ID="${SCENE_IDS[i]}"

    echo "Running preprocessing for $SCENE_ID..."
    python tools/scenefun3d_data_preprocess.py --base_dir "$DATA_BASE" --visit_id "$SCENE_ID" --split "$SPLIT"
done

echo "Batch $BATCH_ID processing complete."