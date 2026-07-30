import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from t_funs3d.utils import io
from t_funs3d.utils.hf_models import (
    extract_points,
    inference_molmo,
    init_molmo,
    init_sam_model,
    process_sam_prompts,
)
from t_funs3d.utils.misc import args2dict, make_frame_data, sort_alphanumeric
from t_funs3d.utils.sun3d.data_parser import DataParser

import time
from tqdm import tqdm


def get_molmo_prompt(visit_id: str, desc_id: str, llm_annot: dict, type: str) -> str:

    original = llm_annot["prompt"]
    func_object = None
    try:
        func_object = llm_annot["acted_on_object"]

        # check that this is not a list
        if isinstance(func_object, list):
            func_object = func_object[0]

        if type == "default":
            prompt = "Points to all the " + func_object.lower()

        elif type == "original":
            prompt = "Points to all the {} in order to {}".format(
                func_object.lower(), original.lower()
            )
        elif type == "original_fixed":
            prompt = "Point to all the {}s in order to {}".format(
                func_object.lower(), original.lower()
            )
    except:
        print(f"Parsing error on {visit_id},{desc_id}. Defaulting to original prompt.")
        prompt = "Points to all the objects in order to " + original.lower()

    ctx_object = None
    try:
        ctx_object = llm_annot["acted_on_object_hierarchy"][0].lower()
    except:
        print(f"No contextual object for {visit_id},{desc_id}.")

    return prompt, func_object, ctx_object


def get_molmo_sam_masks(
    molmo_m, molmo_t, sam_m, sam_t, frame_data, prompt, args
) -> np.ndarray:

    """
    Execute Molmo+Sam masking according to current image and prompt.
    Return N masks obtained by SAM, or None if no objects where found
    """
    img = Image.open(frame_data[0])
    response = inference_molmo(molmo_m, molmo_t, img, prompt)
    h, w = np.asarray(img).shape[:2]

    # points = np.random.randint(200, 1000, size=(3, 2))
    points = extract_points(response, (w, h))

    if points is not None:
        masks = process_sam_prompts(sam_m, sam_t, img, points, batch=5)
        # default setting, each mask counts as one
        scores = np.asarray([1 for _ in masks])

    else:
        masks = None
        scores = None

    return points, masks, scores


def make_object_data(visit_id: str, desc_data: dict, llm_data: dict) -> dict:
    """
    Return dict with data about the presence of each top object in a scene, for each annotation
    """

    obj_data = dict()
    for desc_annot, llm_annot in zip(desc_data, llm_data):
        desc_id = desc_annot["desc_id"]
        ctx_object, _ = io.get_prompt_data(visit_id, desc_id, llm_annot)

        # could be in case of error
        if ctx_object is not None:
            if ctx_object not in obj_data.keys():
                obj_data[ctx_object] = set()
            obj_data[ctx_object].add(desc_id)

    return obj_data


def save_record(path: str, data: dict):
    if data is not None:
        orig_dims = np.asarray([m.shape for m in data["func_masks"]])

        resized_f = list()
        for mask_f in data["func_masks"]:
            if mask_f.shape != (1920, 1440):
                mask_f = torch.tensor(mask_f).unsqueeze(0).unsqueeze(0).to(torch.float)
                mask_f = (
                    torch.nn.functional.interpolate(
                        mask_f, (1920, 1440), mode="nearest"
                    )
                    .squeeze()
                    .numpy()
                    .astype(np.uint8)
                )

            resized_f.append(mask_f)
        print(len(resized_f))

        np.savez_compressed(
            path,
            frame_ids=data["frame_ids"],
            video_ids=data["video_ids"],
            masks_f=np.stack(resized_f, axis=0),
            scores_f=data["func_scores"],
            points=data["points"],
            orig_dims=orig_dims,
        )
    else:
        empty = np.asarray([0])
        np.savez_compressed(
            path,
            frame_ids=empty,
            video_ids=empty,
            masks_f=empty,
            scores_f=empty,
            points=empty,
            orig_dims=empty,
        )


@hydra.main(config_path="../configs", config_name="func_segm")
def molmo_pipeline(args: DictConfig):

    """
    Lifts visual-retrieved SAM masks of a given annotation, and projects them to 3D.
    Returns coalesced 3D masks for each annotationd
    """
    arg_dict = args2dict(args)
    for k, v in arg_dict.items():
        print(f"{k} : {v}")

    # init point cloud parser
    parser = DataParser(args.dataset.root, args.dataset.split)
    molmo_model, molmo_t = init_molmo()
    sam_model, sam_t = init_sam_model("cuda")

    if args.exp_root is None:
        args.exp_root = ""
    if args.exp_name is None:
        args.exp_name = ""

    if args.exp_root != "" or args.exp_name != "":
        os.makedirs(join(args.exp_root, args.exp_name), exist_ok=True)
    
    os.makedirs(join(args.exp_root, args.exp_name, args.frame_folder), exist_ok=True)
    with open(join(args.exp_root, args.exp_name, "config.yaml"), "w") as f:
        OmegaConf.save(args, f)

    parser = DataParser(args.dataset.root, args.dataset.split)
    visits = set(sort_alphanumeric(parser.get_visits()))
    # visits = ["421393"]
    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    print(visits)
    visit_ids = sorted(list(visits))[start:end]

    print(
        f"Running molmo on {end-start} visits (split {args.dataset.split}), from {visit_ids[0]} to {visit_ids[-1]}"
    )

    # iterate over visits
    time_s = time.time()
    desc_count = 0
    for visit_id in tqdm(visit_ids):

        desc_data = parser.get_descriptions(visit_id)
        print([desc["description"] for desc in desc_data])
        # print(desc_data)
        llm_data = parser.get_llm_data(visit_id, args.llm_type)
        # print(llm_data)

        # get all the data and frames for all the objects the descs of this scene!
        obj_data = make_object_data(visit_id, desc_data, llm_data)
        print(obj_data)
        obj_frames_data = dict()
        for obj_id in obj_data.keys():
            print(obj_id)
            for desc_id in obj_data[obj_id]:
                print(f" - {desc_id}")
                # pre-loads all context object objects of this visit
                obj_frames_data[desc_id] = io.sample_scored_frames(
                    parser,
                    visit_id,
                    desc_id,
                    args.frame_sampling.n,
                    args.mask_type,
                    os.path.join(args.exp_root, args.exp_name, "association"),
                )
                print(f"Object {obj_id} has {len(obj_frames_data[desc_id]['rgb_paths'])} frames")
        print(obj_frames_data.keys())

        print("We run molmo here")
        for desc_annot, llm_annot in zip(desc_data, llm_data):
            desc_count += 1
            desc_id = desc_annot["desc_id"]
            # print("We run molmo here")

            (molmo_prompt, func_object, ctx_object) = get_molmo_prompt(
                visit_id, desc_id, llm_annot, args.molmo_prompt
            )

            base_path = join(args.exp_root, args.exp_name, args.frame_folder, visit_id)
            desc_path = base_path + "_" + desc_id + ".npz"
            if os.path.exists(desc_path):
                print(
                    "Prediction {},{} exists in {}. Skipping.".format(
                        visit_id, desc_id, args.exp_name
                    )
                )
                continue

            mask_list = list()
            score_list = list()
            point_list = list()
            mask_n = list()

            # contextual object must be valid, otherwise there is an error, i.e. and empty mask.
            if ctx_object is not None:
                print(f"Processing {visit_id},{desc_id} with context {ctx_object}.")
                molmo_frames = obj_frames_data[desc_id]

                for frame_data in tqdm(zip(*molmo_frames.values()), total=len(next(iter(molmo_frames.values()))), desc=f"Processing frames for {ctx_object}"):

                    points, masks, scores = get_molmo_sam_masks(
                        molmo_model,
                        molmo_t,
                        sam_model,
                        sam_t,
                        frame_data,
                        molmo_prompt,
                        args,
                    )

                    if masks is not None:
                        mask_n.append(masks.shape[0])
                    else:
                        mask_n.append(0)
                        points = None

                    mask_list.append(masks)
                    score_list.append(scores)
                    point_list.append(points)
                    # first is always image
                # print(mask_list)
                # print(score_list)
                print(mask_n)

                # add info about functional object mask
                molmo_frames = make_frame_data(
                    molmo_frames, mask_list, mask_n, score_list, point_list, args
                )
                # if non empty, save record to be used later
                print(molmo_frames["frame_ids"].shape[0])
                if molmo_frames["frame_ids"].shape[0] > 0:
                    save_record(desc_path, molmo_frames)
                else:
                    save_record(desc_path, None)
            else:
                save_record(desc_path, None)
    print(f"[INFO] Totol runtime: {time.time() - time_s:.2f}s.")
    print(f"[INFO] Average runtime: {(time.time() - time_s)/desc_count:.2f}s per tasks.")


if __name__ == "__main__":

    molmo_pipeline()
