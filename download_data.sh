#!/bin/bash
# Download required checkpoint and example asset files for the T-FunS3D project.
# This script creates a checkpoint directory, fetches model weights and sample data,
# and places them in the expected workspace locations.

mkdir checkpoints && cd checkpoints

# Download the Mask3D pre-trained model checkpoints for ScanNet200
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B&export=download&authuser=0&confirm=t&uuid=7242e4da-024f-4cb9-b3ae-fb12bb1e022e&at=AN8xHoorFAQZ_SBG903qLxKUbJMP%3A1750172923367" -O scannet200_val.ckpt
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y&export=download&authuser=0&confirm=t&uuid=33897c2a-8be1-4283-b8d6-ff1d6a94da4d&at=AN8xHoqCLPLbhQELU_s5JZL7Y2fq%3A1750173092892" -O scannet200_model.ckpt
# Download the pre-trained model checkpoint for the SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download the example scene data for testing the T-FunS3D pipeline
cd ..
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1UOwBZMCrTMg-_MFwmYkKOrex1YS6Nw-i&export=download&authuser=0&confirm=t&uuid=88fda28a-aa3b-4912-8b25-28299f81788f&at=AN8xHopGCcgLfxlD0UQ4INesSlN-%3A1750174758521" -O scene_example.zip
