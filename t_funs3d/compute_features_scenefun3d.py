import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from t_funs3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from t_funs3d.utils.misc import get_free_gpu, create_out_folder
from t_funs3d.features_computation.features_extractor import FeaturesExtractor
import torch
import json
from glob import glob
from tqdm import tqdm
import time

# TIP: add version_base=None to the arguments if you encounter some error  
@hydra.main(config_path="configs", config_name="openmask3d_inference")
def main(ctx: DictConfig):
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu(7000) if torch.cuda.is_available() else device
    print(f"[INFO] Using device: {device}")
    out_folder = ctx.output.output_directory
    os.chdir(hydra.utils.get_original_cwd())
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(os.path.join(ctx.output.output_directory, "config.yaml"), "w") as f:
        OmegaConf.save(ctx, f)
        
    masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)))
    start = 0 if ctx.dataset.start is None else int(ctx.dataset.start)
    end = len(masks_paths) if ctx.dataset.end is None else int(ctx.dataset.end)
    masks_paths = masks_paths[start:end]
    print(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)
    print(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix))

    root_path = ctx.dataset.root
    data_split = ctx.dataset.split
    scene_dir = os.path.join(root_path, data_split)

    times_per_scene = []
    results_str = ""
    
    for masks_path in tqdm(masks_paths):
        assert os.path.exists(masks_path), f"Path to masks does not exist: {masks_path} - first run compute_masks_single_scene.sh!"

        scene_name = masks_path.split('/')[-1][:6]
        scene_path = os.path.join(scene_dir, scene_name)
        point_cloud_path = os.path.join(scene_path, f"{scene_name}_laser_scan_downsampled.ply")
        print(f"[INFO] Computing feature of {scene_name}")

        # Get all name of folders in the path
        video_paths = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))])
        print(f"[INFO] Saving feature results to {out_folder}")
        out_folder_scene = os.path.join(out_folder, scene_name)
        if not os.path.exists(out_folder_scene):
            os.makedirs(out_folder_scene)

        print(f"[INFO] Found video folders: {video_paths}")
        times_per_video = []
        for video_path in video_paths:

            time_start_video = time.time()

            video_id = video_path.split('/')[-1]
            
            poses_path = os.path.join(scene_path, video_id, ctx.data.camera.poses_path)
            
            intrinsic_path = os.path.join(scene_path, video_id, ctx.data.camera.intrinsic_path)
            images_path = os.path.join(scene_path, video_id, ctx.data.images.images_path)
            depths_path = os.path.join(scene_path, video_id, ctx.data.depths.depths_path)
            
            # 1. Load the masks
            masks = InstanceMasks3D(masks_path)
            print(f"[INFO] Masks loaded. {masks.num_masks} masks found.")

            # 2. Load the images
            if ctx.data.dataset == "scenefun3d":
                # Load indicis from timestamps
                ts_path = os.path.join(os.path.dirname(poses_path), "timestamps.txt")
                with open(ts_path, 'r') as f:
                    # Read lines and strip whitespace, read one line every ctx.openmask3d.frequency
                    indices = [l.strip() for i, l in enumerate(f.readlines()) if i % ctx.openmask3d.frequency == 0]
            else: # scannet
                indices = np.arange(0, get_number_of_images(poses_path), step=ctx.openmask3d.frequency)
            
            images = Images(images_path=images_path, 
                                extension=ctx.data.images.images_ext, 
                                indices=indices)
            print(f"[INFO] Images loaded. {len(images.images)} images found.")

            # 3. Load the pointcloud
            pointcloud = PointCloud(point_cloud_path)
            print(f"[INFO] Pointcloud loaded. {pointcloud.num_points} points found.")

            # 4. Load the camera configurations
            camera = Camera(intrinsic_path=intrinsic_path, 
                            intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                            poses_path=poses_path, 
                            depths_path=depths_path, 
                            extension_depth=ctx.data.depths.depths_ext, 
                            depth_scale=ctx.data.depths.depth_scale)
            print("[INFO] Camera configurations loaded.")

            # 5. Run extractor
            print("[INFO] Computing per-mask CLIP features.")
            features_extractor = FeaturesExtractor(camera=camera, 
                                                    clip_model=ctx.external.clip_model, 
                                                    images=images, 
                                                    masks=masks,
                                                    pointcloud=pointcloud, 
                                                    sam_model_type=ctx.external.sam_model_type,
                                                    sam_checkpoint=ctx.external.sam_checkpoint,
                                                    vis_threshold=ctx.openmask3d.vis_threshold,
                                                    device=device)

            features, mask_view_association = features_extractor.extract_features(topk=ctx.openmask3d.top_k, 
                                                            multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,
                                                            num_levels=ctx.openmask3d.num_of_levels, 
                                                            num_random_rounds=ctx.openmask3d.num_random_rounds,
                                                            num_selected_points=ctx.openmask3d.num_selected_points,
                                                            save_crops=ctx.output.save_crops,
                                                            out_folder=out_folder_scene,
                                                            optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
            print("[INFO] Features computed.")
            times_per_video.append(time.time() - time_start_video)
            print(f"[INFO] Time taken for video {video_id} with {len(images.images)} frames: {times_per_video[-1]:.2f} seconds")
            # accumulate results to a string
            results_str += f"Scene: {scene_name}, Video: {video_id}, {len(images.images)} frames, Time: {times_per_video[-1]:.2f} seconds\n"
            
            # 6. Save features
            filename = f"{scene_name}_{video_id}_features.npy"
            output_path = os.path.join(out_folder_scene, filename)
            np.save(output_path, features)
            print(f"[INFO] Mask features for scene {scene_name} and video {video_id} saved to {output_path}.")

            # 7. Save mask view association
            ass_name = f"{scene_name}_{video_id}_association.json"
            ass_outpath = os.path.join(out_folder_scene, ass_name)
            with open(ass_outpath, 'w') as f:
                json.dump(to_json_safe(mask_view_association), f, ensure_ascii=False, separators=(',', ': '), indent=4)
            print(f"[INFO] Mask view association for scene {scene_name} and video {video_id} saved to {ass_outpath}.")
        times_per_scene.append(np.mean(times_per_video))
    print(f"[INFO] Average time per scene: {np.mean(times_per_scene):.2f} seconds")
    results_str += f"Average time per scene: {np.mean(times_per_scene):.2f} seconds\n"
    # Save results to a text file
    results_path = os.path.join(out_folder, "time_profile.txt")
    with open(results_path, 'w') as f:
        f.write(results_str)
    print(f"[INFO] Results saved to {results_path}")
    

def to_json_safe(obj):
    """Recursively transform the data structure to ensure it can be safely stored as JSON, while keeping mask_id/view_id as int."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert key to str (required by JSON), but if the key is a numpy.int, convert it to Python int first
            if isinstance(k, (np.integer,)):
                k = int(k)
            elif isinstance(k, (np.floating,)):
                k = float(k)
            else:
                k = str(k)  # Other cases are converted to strings.

            new_dict[k] = to_json_safe(v)
        return new_dict

    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    else:
        return obj

if __name__ == "__main__":
    main()