import logging
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from openmask3d.class_agnostic_mask_computation.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer
import open3d as o3d
import numpy as np
import torch
import time
import pdb
from glob import glob

def get_parameters(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #loggers = []

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    #logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, None #loggers


def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals

def process_file(filepath):
    coords, colors, normals = load_ply(filepath)
    raw_coordinates = coords.copy()
    raw_colors = (colors*255).astype(np.uint8)
    raw_normals = normals

    features = colors
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    filename = filepath.split("/")[-1][:-4]
    return [[coords, features, [], filename, raw_colors, raw_normals, raw_coordinates, 0]] # 2: original_labels, 3: none
    # coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], raw_color, raw_normals, raw_coordinates, idx

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_scenefun3d.yaml")
def get_class_agnostic_masks(cfg: DictConfig):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    c_fn = hydra.utils.instantiate(cfg.data.test_collation) #(model.config.data.test_collation)

    root_path = cfg.general.dataset.root
    data_split = cfg.general.dataset.split
    scene_dir = os.path.join(root_path, data_split)

    times = []
    results_str = ""

    # scene_paths = sorted([d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))])
    scene_paths = sorted(glob(os.path.join(scene_dir, "*/*downsampled.ply")))

    start = 0 if cfg.dataset.start is None else int(cfg.dataset.start)
    end = len(scene_paths) if cfg.dataset.end is None else int(cfg.dataset.end)
    scene_paths = scene_paths[start:end]

    for scene_path in scene_paths:
        input_batch = process_file(scene_path)
        batch = c_fn(input_batch)

        model.to(device)
        model.eval()

        start = time.time()
        with torch.no_grad():
            res_dict = model.get_masks_single_scene(batch)
        end = time.time()
        times.append(end - start)
        print(f"time elapsed: {times[-1]:.2f} seconds")
        results_str += f"Scene: {scene_path}, Time: {times[-1]:.2f} seconds\n"
    
    print(f"Average time per scene: {np.mean(times):.2f} seconds")
    results_str += f"Average time per scene: {np.mean(times):.2f} seconds\n"
    # Save results to a text file
    results_path = os.path.join(cfg.general.mask_save_dir, "time_profile.txt")
    with open(results_path, 'w') as f:        
        f.write(results_str)
    print(f"[INFO] Results saved to {results_path}")
        

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks(cfg)
    
    with open(os.path.join(cfg.general.mask_save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    main()
