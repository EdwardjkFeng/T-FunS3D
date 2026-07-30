import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())
import argparse
import json
import math
from os.path import join
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from t_funs3d.utils import io
from t_funs3d.utils.hf_models import init_detection, process_detection, init_clip_textual
from t_funs3d.utils.misc import sort_alphanumeric
from t_funs3d.utils.sun3d.data_parser import DataParser
import torch
import numpy as np

@hydra.main(config_path="configs", config_name="functionality_segm")
def make_mask_index(args: DictConfig):
    """
    Query the open vocabulary scene graph for each scene and save the association between descriptions and masks.
    """

    root, split = args.dataset.root, args.dataset.split
    start, end = args.dataset.start, args.dataset.end
    parser = DataParser(root, split)

    visits2videos = io.get_visit_to_videos(root, split)
    visits = set(sort_alphanumeric(parser.get_visits()))
    # visits = ["421393"]
    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)
    print(visits)
    visit_ids = sorted(list(visits))[start:end]

    print(
        f"Processing {end-start} visits (split {args.dataset.split}), from {visit_ids[0]} to {visit_ids[-1]}"
    )

    clip_m, clip_t = init_clip_textual("qihoo360/fg-clip-large")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.exp_root is None:
        args.exp_root = ""
    if args.exp_name is None:
        args.exp_name = ""

    # visit_ids = ["421393"]
    os.makedirs(
        os.path.join(args.exp_root, args.exp_name, "association"), exist_ok=True
    )

    for visit_id in tqdm(visit_ids):
        video_list = io.get_visit_to_videos(root, split)[visit_id]
        visit_dict = {"desc_ids": {}, "objects": {}}

        # Load description list
        desc_data = parser.get_descriptions(visit_id)
        llm_data = parser.get_llm_data(visit_id, args.llm_type)
        features = []

        visit_dict = {"desc_ids": {}, "objects": {}}
        for video_id in video_list:
            features.append(parser.get_mask_features(visit_id, 
                                                     video_id, 
                                                     os.path.join(args.exp_root, args.exp_name, "mask_features")))
            masks_frames = parser.get_masks_frames_association(visit_id, 
                                                               video_id, 
                                                               os.path.join(args.exp_root, args.exp_name, "association")
                                                               )["masks"]
            for mask_id, frames in masks_frames.items():
                if mask_id not in visit_dict['objects']:
                    visit_dict['objects'][mask_id] = []
                visit_dict['objects'][mask_id] += list(frames.keys())
        features = np.array(features)
        # print(f"visit_dict: {visit_dict}")

        print(f"Features shape: {features.shape}")
        features = np.mean(features.squeeze(), axis=0)
        print(f"Features shape: {features.shape}")
    
        for desc_dict, cot in zip(desc_data, llm_data):
            desc = desc_dict['description']
            desc_id = desc_dict['desc_id']
            print(desc)
            # ref_object = cot['referent_object_hierarchy']
            relations = [rel.replace("_", " ") for rel in cot['spatial_relation']]
            print(relations)
            rel_object = cot['referent_object_hierarchy']
            cxt_object = cot['acted_on_object_hierarchy'][0]
            print(cxt_object)
            # object_set = set(ref_object)
            # object_set.add(cxt_object)
            object_set = relations
            object_set += rel_object
            object_set.append(cxt_object)
            print(object_set)
            text_input_processed = clip_t(list(object_set), padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_embeddings = clip_m.get_text_features(**text_input_processed, walk_short_pos=True)
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            print(text_embeddings.shape)
            # text_input_processed = clip_t(list(object_set)).to(device)
            # with torch.no_grad():
            #     text_embeddings = clip_m.encode_text(text_input_processed)
            # print(text_embeddings.shape)

            sim_scores = text_embeddings.float().cpu() @ features.T
            sim_scores = clip_m.logit_scale.exp().detach().cpu() * sim_scores
            sim_scores = sim_scores.softmax(dim=1).numpy()
            print(sim_scores)
            # Get the indices of the top n masks of each row
            top_n_indices = sim_scores.argsort(axis=1)[:, -2:]
            # print(top_n_indices)
            # top_n_indices = set(top_n_indices.flatten().tolist())
            selected = set()
            for idx, scores in zip(top_n_indices, sim_scores):
                for i in idx:
                    if scores[i] > 0.1:
                        selected.add(i)

            print(selected)
            top_n_indices = selected
            print(f"Top n indices: {top_n_indices}")

            if cxt_object not in visit_dict['desc_ids'].keys():
                # visit_dict['desc_ids'][cxt_object] = list()
                visit_dict['desc_ids'][desc_id] = list()
            
            for mask_id in top_n_indices:
                # print(visit_dict['desc_ids'][cxt_object])
                # print(visit_dict['desc_ids'][desc_id])
                # print(visit_dict['objects'][str(mask_id)])
                # visit_dict['desc_ids'][cxt_object] += visit_dict['objects'][str(mask_id)]
                visit_dict['desc_ids'][desc_id] += visit_dict['objects'][str(mask_id)]
            # visit_dict['desc_ids'][cxt_object] = list(set(visit_dict['desc_ids'][cxt_object]))
            visit_dict['desc_ids'][desc_id] = list(set(visit_dict['desc_ids'][desc_id]))

            print("-----")

        with open(
            os.path.join(
                # root, split, visit_id, f"{visit_id}_{args.mask_type}_masks.json"
                args.exp_root, args.exp_name, f"association/{visit_id}_{args.mask_type}_masks.json"
            ),
            "w",
        ) as f:
            json.dump(visit_dict, f)

if __name__ == "__main__":
    make_mask_index()