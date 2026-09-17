#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export HF_HOME="$(pwd)/cache/" # Specify cache dir for daic compute nodes
set -e

# RUN OPENMASK3D FOR A BATCH OF SCENES of the SceneFun3D dataset
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths
ROOT="$(pwd)/datasets/scenefun3d"
SPLIT="val"
START=0
END=1

SCENE_POSE_DIR="processed/pose"
SCENE_INTRINSIC_PATH="hires_wide_intrinsics"
SCENE_INTRINSIC_RESOLUTION="[1440,1920]" # change if your intrinsics are based on another resolution
SCENE_COLOR_IMG_DIR="hires_wide"
SCENE_DEPTH_IMG_DIR="hires_depth"
IMG_EXTENSION=".jpg"
DEPTH_EXTENSION=".png"
DEPTH_SCALE=1000
# model ckpt paths
MASK_MODULE_CKPT_PATH="$(pwd)/checkpoints/scannet200_model.ckpt"
SAM_CKPT_PATH="$(pwd)/checkpoints/sam_vit_h_4b8939.pth"

# output directories to save masks and mask features
EXPERIMENT_NAME="eval"

OUTPUT_DIRECTORY="$(pwd)/t-funs3d_outputs" 
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/${DATE}/${TIME}"
MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
SAVE_VISUALIZATIONS=true #if set to true, saves pyviz3d visualizations
SAVE_CROPS=true 
# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd t_funs3d

# 1. Compute class agnostic masks and save them
echo "[INFO] Extracting class agnostic masks..."
python get_masks_scenefun3d.py \
general.experiment_name=${EXPERIMENT_NAME} \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
data.test_mode=test \
model.num_queries=120 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.dbscan_min_points=50 \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
general.save_dir=${MASK_SAVE_DIR} \
general.mask_save_dir=${MASK_SAVE_DIR} \
general.filter_out_instances=true \
general.scores_threshold=0.1 \
general.iou_threshold=0.2 \
+general.dataset.root=${ROOT} \
+general.dataset.split=${SPLIT} \
+dataset.start=${START} \
+dataset.end=${END} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}" 
echo "[INFO] Mask computation done!"

# get the path of the saved masks
MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}/masks"
echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

# 2. Compute mask features for each mask and save them
echo "[INFO] Computing mask features..."

python compute_features_scenefun3d.py \
data.masks.masks_path=${SCENE_MASK_PATH} \
data.camera.poses_path=${SCENE_POSE_DIR} \
data.camera.intrinsic_path=${SCENE_INTRINSIC_PATH} \
data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
data.depths.depths_path=${SCENE_DEPTH_IMG_DIR} \
data.depths.depth_scale=${DEPTH_SCALE} \
data.depths.depths_ext=${DEPTH_EXTENSION} \
data.images.images_path=${SCENE_COLOR_IMG_DIR} \
data.images.images_ext=${IMG_EXTENSION} \
output.output_directory=${MASK_FEATURE_SAVE_DIR} \
output.save_crops=${SAVE_CROPS} \
openmask3d.vis_threshold=0.1 \
+dataset.root=${ROOT} \
+dataset.split=${SPLIT} \
+dataset.start=${START} \
+dataset.end=${END} \
+data.dataset="scenefun3d" \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}" \
external.sam_checkpoint=${SAM_CKPT_PATH} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE}
echo "[INFO] Feature computation done!"

MASK_TYPE="standard"
LLM_TYPE="Qwen3-8B"


# Stage II: Task-driven 3d functionality segmentation
# 3. Parse task descriptions and decompose them into ontology and functionality.
echo "[INFO] Parsing task descriptions and decomposing them into ontology and functionality..."

python parse_task_descriptions.py dataset.root=$ROOT dataset.split=$SPLIT llm_type=$LLM_TYPE hydra.run.dir=$OUTPUT_FOLDER_DIRECTORY dataset.start=$START dataset.end=$END exp_root=$OUTPUT_FOLDER_DIRECTORY
echo "[INFO] Task description parsing and decomposition for ${START} - ${END} done!"


# 4. Query open vocabulary scene graph for each scene and save the results
echo "[INFO] Querying open vocabulary scene graph for candidate nodes..."

python make_masks_index.py dataset.root=$ROOT dataset.split=$SPLIT mask_type=$MASK_TYPE llm_type=$LLM_TYPE hydra.run.dir=$OUTPUT_FOLDER_DIRECTORY dataset.start=$START dataset.end=$END exp_root=$OUTPUT_FOLDER_DIRECTORY
echo "[INFO] Scene graph querying for ${START} - ${END} done!"


# 5. Run MolMO to generate functionality masks for each task and save them
echo "[INFO] Running Molmo to generate functionality masks for each task..."

python functionality_segmentation/run_molmo_sam.py dataset.root=$ROOT dataset.split=$SPLIT mask_type=$MASK_TYPE llm_type=$LLM_TYPE hydra.run.dir=$OUTPUT_FOLDER_DIRECTORY dataset.start=$START dataset.end=$END exp_root=$OUTPUT_FOLDER_DIRECTORY
echo "[INFO] Molmo for ${START} - ${END} done!"


# 6. Lift the functionality masks to 3D and save them
echo "[INFO] Lifting the functionality masks to 3D..."

python functionality_segmentation/run_lifting.py dataset.root=$ROOT dataset.split=$SPLIT mask_type=$MASK_TYPE llm_type=$LLM_TYPE hydra.run.dir=$OUTPUT_FOLDER_DIRECTORY dataset.start=$START dataset.end=$END exp_root=$OUTPUT_FOLDER_DIRECTORY
echo "[INFO] Lifting for ${START} - ${END} done!"


# Evaluation
echo "[INFO] Evaluating for ${START} - ${END}..."

python evaluation/evaluate.py dataset.root=$ROOT dataset.split=$SPLIT mask_type=$MASK_TYPE llm_type=$LLM_TYPE hydra.run.dir=$OUTPUT_FOLDER_DIRECTORY dataset.start=$START dataset.end=$END exp_root=$OUTPUT_FOLDER_DIRECTORY
echo "[INFO] Evaluation for ${START} - ${END} done!"