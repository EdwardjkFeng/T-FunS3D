import os
import sys

sys.path.append(os.getcwd())
from os.path import join
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray

from t_funs3d.utils.misc import np_normalize
from t_funs3d.utils.sun3d.data_parser import DataParser


def get_prompt_data(visit_id: str, desc_id: str, llm_annot: dict) -> Tuple[str, str]:
    """
    Returns context and functional object
    """

    context_object, func_object = None, None  # these are top object and functional
    try:
        context_object = llm_annot["acted_on_object_hierarchy"][0].lower()
        func_object = llm_annot["acted_on_object"].lower()

    except:
        print(f"No contextual or func object for {visit_id},{desc_id}.")

    return context_object, func_object

def sample_scored_frames(
    parser: DataParser,
    visit_id: str,
    obj: str,
    n_samples: int,
    mask_type: str,
    exp_dir: str = None,
) -> dict:

    """
    Samples N images across all views for a visit with a specified object.
    """

    cand_videos, cand_frames = list(), list()
    # query_object = obj.lower()
    query_object = obj
    mask_data = parser.get_mask_index(visit_id, mask_type, exp_dir)

    if query_object in mask_data["desc_ids"].keys():
        obj_set = set(mask_data["desc_ids"][query_object])

        # list of video_id/frame_id pairw with the given object with the given description
        video_frames_ids = list(obj_set)
        print(f"Object {obj} found in {len(video_frames_ids)} frames.")

        for video_frame_id in video_frames_ids:

            video_id, frame_id = video_frame_id.split("_")
            cand_videos.append(video_id)
            cand_frames.append(frame_id)

    cand_videos = np.asarray(cand_videos)
    cand_frames = np.asarray(cand_frames)

    if cand_videos.shape[0] == 0:
        return {
            "rgb_paths": np.asarray([]),
            "depth_paths": np.asarray([]),
            "video_ids": np.asarray([]),
            "frame_ids": np.asarray([]),
            "intrinsics": np.asarray([]),
            "poses": np.asarray([]),
        }

    sampling = False
    # print(cand_videos,cand_frames,cand_boxes)
    modality = None
    if modality is not None and modality == "random":
        sampling = True
        actual_samples = min(cand_videos.shape[0], n_samples)
        sub_idxs = np.random.choice(
            np.arange(cand_videos.shape[0]), actual_samples, replace=False
        )
        cand_videos = cand_videos[sub_idxs]
        cand_frames = cand_frames[sub_idxs]

    else:
        sampling = False

    # list of selected samples
    sel_rgb, sel_depth, sel_intrinsics, sel_poses = (list(), list(), list(), list())

    # dictionaries of data for each video
    poses_t, rgb_paths_t, depth_paths_t, intrinsics_t = (dict(), dict(), dict(), dict())
    for video_id in np.unique(cand_videos):

        # save video-related information in dictionary
        poses_t[video_id] = parser.get_camera_trajectory(visit_id, video_id)
        rgb_paths_t[video_id] = parser.get_rgb_frames(visit_id, video_id)
        depth_paths_t[video_id] = parser.get_depth_frames(visit_id, video_id)
        intrinsics_t[video_id] = parser.get_camera_intrinsics(visit_id, video_id)

    for frame_id, video_id in zip(cand_frames, cand_videos):

        sel_rgb.append(rgb_paths_t[video_id][frame_id])
        sel_depth.append(depth_paths_t[video_id][frame_id])
        intrinsic = parser.read_camera_intrinsics(
            intrinsics_t[video_id][frame_id], format="matrix"
        )
        sel_intrinsics.append(intrinsic)
        sel_poses.append(parser.get_nearest_pose(frame_id, poses_t[video_id]))

    # if sampling:
    #     cand_masks = [all_masks[idx] for idx in sub_idxs]
    # else:
    #     cand_masks = all_masks

    # necessary to get all boxes

    return {
        "rgb_paths": np.asarray(sel_rgb),
        "depth_paths": np.asarray(sel_depth),
        "video_ids": cand_videos,
        "frame_ids": cand_frames,
        "intrinsics": np.asarray(sel_intrinsics),
        "poses": np.asarray(sel_poses),
        # "masks": cand_masks,
    }

def filter_scored_masks(
    parser: DataParser, visit_id: str, mask_type: str, query_object: str, exp_dir: str = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Filters scored masks by only retaining frames with a specific object
    """
    video_ids, frame_ids = list(), list()
    query_object = query_object.lower()
    mask_data = parser.get_mask_index(visit_id, mask_type, exp_dir)

    if query_object in mask_data["objects"].keys():
        obj_set = set(mask_data["objects"][query_object])

        # list of video_id/frame_id pairw with the given object with the given description
        video_frames_ids = list(obj_set)

        for video_frame_id in video_frames_ids:

            video_id, frame_id = video_frame_id.split(" ")
            video_ids.append(video_id)
            frame_ids.append(frame_id)

    video_ids = np.asarray(video_ids)
    frame_ids = np.asarray(frame_ids)

    return video_ids, frame_ids


def get_visit_to_videos(root: str, split: str) -> dict:
    """
    Given a split, returns a dict associating each visit id to the list of video ids
    """

    visit_to_videos = dict()

    with open(join(root, f"benchmark_file_lists/{split}_set.csv")) as f:
        # skip csv header
        visit_video = f.readlines()[1:]

    for line in visit_video:
        visit_id, video_id = line.strip("\n").split(",")
        if visit_id not in visit_to_videos.keys():
            visit_to_videos[visit_id] = list()
        visit_to_videos[visit_id].append(video_id)

    return visit_to_videos


def pad_detection_predictions(
    boxes: List, scores: List, labels: List
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    List of lists of GDino prediction for each frame.
    Pads them to the largest value. Non-valid detections are set with scores=-1
    """

    assert len(boxes) == len(scores) and len(scores) == len(labels)
    num_frames = len(boxes)
    max_boxes = max([len(frame_boxes) for frame_boxes in boxes])

    new_boxes = np.zeros((num_frames, max_boxes, 4))
    new_scores = np.zeros((num_frames, max_boxes))
    new_labels = list()
    for i, (boxes_i, scores_i, labels_i) in enumerate(zip(boxes, scores, labels)):
        assert len(boxes_i) == len(scores_i) and len(scores_i) == len(labels_i)

        if len(labels_i) < max_boxes:
            for _ in range(max_boxes - len(labels_i)):
                labels_i.append("empty")

        boxes_i, scores_i, labels_i = (
            np.asarray(boxes_i),
            np.asarray(scores_i),
            np.asarray(labels_i),
        )
        num_boxes = boxes_i.shape[0]
        new_boxes[i, :num_boxes] = boxes_i
        new_scores[i, :num_boxes] = scores_i
        new_labels.append(labels_i)

    new_labels = np.asarray(new_labels)
    return new_boxes.astype(np.uint16), new_scores, new_labels
