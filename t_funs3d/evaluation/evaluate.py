import os
import sys

import torch

sys.path.append(os.getcwd())
from os.path import join

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from openmask3d.utils import io
from openmask3d.utils.evaluator import Segment3DEvaluator
from openmask3d.utils.misc import np_normalize, sort_alphanumeric
from openmask3d.utils.sun3d.data_parser import DataParser


def viz_3d_masks(pcd: np.array, gt_mask: np.array, pred_mask: np.array, outfile: str) -> None:
    """
    Visualizes predicted masks agains a ground truth. Keys:
    Masks are represented as binary masks
    - gt mask only is visualized in blue
    - pred mask only is visualized in red
    - overlap part is visualized in green
    """

    assert (pcd.shape[0] == gt_mask.shape[0]) and (
        pred_mask.shape[0] == gt_mask.shape[0]
    ), " Mask size do not correspond."

    # TODO: assert that colors are in range 0..1 as required by open3d

    xyz, rgb = pcd[:, :3].copy(), pcd[:, 3:].copy()

    # compute overlap masks
    overlap = np.where(np.logical_and(gt_mask, pred_mask), 1.0, 0.0)

    gt_idx = np.nonzero(gt_mask)[0]
    pred_idx = np.nonzero(pred_mask)[0]
    overlap_idx = np.nonzero(overlap)[0]

    # ground truth only in blue
    rgb[gt_idx] = np.asarray([0.0, 0.0, 1.0])
    # pred only in red
    rgb[pred_idx] = np.asarray([1.0, 0.0, 0.0])
    # # # commmon (IoU) in green
    if overlap_idx.shape[0] != 0:
        rgb[overlap_idx] = np.asarray([0.0, 1.0, 0.0])

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # The size of the axes, adjust based on your scene
        origin=[0, 0, 0],  # The origin of the coordinate frame
    )

    # o3d.visualization.draw_geometries([o3d_pcd])
    o3d.io.write_point_cloud(outfile, o3d_pcd)
    


def get_present_annots(path: str) -> dict:
    """
    Returns dict of present visits -> list of annot.
    To be used with the experiment path.
    """

    visit2desc = dict()
    for file in os.listdir(join(path, "pcds")):

        visit_id, desc_id = os.path.splitext(file)[0].split("_")
        if visit_id not in visit2desc.keys():
            visit2desc[visit_id] = list()
        visit2desc[visit_id].append(desc_id)

    return visit2desc


def post_process_pcd(
    pcd: np.ndarray,
    masks: dict,
    args: DictConfig,
) -> torch.Tensor:
    """
    Converts a point-cloud heatmap in a binary point cloud mask according to evaluation modality.
    """
    n_views = masks["n_views"]
    acc_f = masks["acc_f"]
    if acc_f.sum() == 0:
    #     return torch.zeros(acc_f.shape[0])
        print("[Warning] acc_f = 0")
    p_f = np_normalize(acc_f / n_views)

    pred_mask = np.where(p_f > args.threshold, 1, 0)

    return torch.tensor(pred_mask)


@hydra.main(config_path="../configs", config_name="func_segm")
def main(args: DictConfig):

    evaluate_molmo(args)


def evaluate_molmo(args: DictConfig) -> dict:
    """
    Evaluate 3D masks against ground-truth
    """

    if args.exp_root is None:
        args.exp_root = ""
    if args.exp_name is None:
        args.exp_name = ""
    os.makedirs(join(args.exp_root, args.exp_name, f"viz_{args.threshold}"), exist_ok=True)
    
    exp_tag = f"{args.exp_name} {args.pcds_folder}_($>${args.threshold})"
    evaluator = Segment3DEvaluator(exp_tag)
    parser = DataParser(args.dataset.root, args.dataset.split)
    # visits = io.get_visit_to_videos(args.dataset.root, args.dataset.split).keys()
    visits = set(sort_alphanumeric(parser.get_visits()))
    # visits = ["421393"]
    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    print(visits)
    visits = sorted(list(visits))[start:end]

    for visit_id in tqdm(visits):
        print(visit_id)
        # get cropped pcd and move it to np
        pcd = parser.get_laser_scan(visit_id)
        pcd = parser.get_cropped_laser_scan(visit_id, pcd)

        scene_xyz = np.asarray(pcd.points)
        scene_rgb = np.asarray(pcd.colors)
        full_pcd = np.concatenate([scene_xyz, scene_rgb], axis=1)
        #llm_data = parser.get_llm_data(visit_id, llm=args.llm_type)
        desc_data = parser.get_descriptions_list(visit_id)

        os.makedirs(join(args.exp_root, args.exp_name, f"viz_{args.threshold}", visit_id), exist_ok=True)

        # iterate over annotations of this desc_id
        for desc_id in desc_data.keys():
        #for (desc_id, desc_prompt), llm_annot in zip(desc_data.items(), llm_data):

            gt_mask = parser.get_grouped_annotation(visit_id, desc_id)
            gt_mask = torch.tensor(gt_mask)

            # NB: predicted mask refer to CROPPED point cloud
            path = join(
                args.exp_root,
                args.exp_name,
                args.pcds_folder,
                f"{visit_id}_{desc_id}.npz",
            )
            if os.path.exists(path):
                mask_data = np.load(path)
                pred_mask = post_process_pcd(full_pcd, mask_data, args)
            else:
                print(
                    "Predicted mask for {},{} does not exist, setting it to empty mask.".format(
                        visit_id, desc_id
                    )
                )
                pred_mask = torch.zeros(full_pcd.shape[0])

            if pred_mask.sum() == 0:
                print(f"[Warning] Predicted mask for {visit_id},{desc_id} does not exist, setting it to empty mask.")
                continue
                
            evaluator.register([visit_id], [desc_id], gt_mask, pred_mask)

            if "viz" in args:
                viz_3d_masks(
                    full_pcd,
                    gt_mask.numpy(),
                    pred_mask.numpy(),
                    f"{args.exp_root}/{args.exp_name}/viz_{args.threshold}/{visit_id}/results_pcd_{desc_id}.ply",
                )

    print(evaluator.get_latex_str())

    if "save_results" in args:
        with open(
            f"{args.exp_root}/{args.exp_name}/results_pcds_{args.threshold}.json", "w"
        ) as f:
            evaluator.save(f)


if __name__ == "__main__":

    main()
