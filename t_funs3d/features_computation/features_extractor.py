import os
from PIL import Image
import cv2
import clip
import numpy as np
import imageio
import torch
from tqdm import tqdm
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
# from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
from openmask3d.mask_features_computation.utils import *
from openmask3d.utils.hf_models import init_clip_visual, init_sam_model, process_sam_points

from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
import time

class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: InstanceMasks3D, 
                 vis_threshold, 
                 indices):
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        
        
    # def get_visible_points_view(self):
    #     # Initialization
    #     vis_threshold = self.vis_threshold
    #     indices = self.indices
    #     depth_scale = self.camera.depth_scale
    #     poses = self.camera.load_poses(indices)
    #     X = self.point_cloud.get_homogeneous_coordinates()
    #     n_points = self.point_cloud.num_points
    #     depths_path = self.camera.depths_path        
    #     resolution = imageio.imread(os.path.join(depths_path, f"{indices[0]}.png")).shape
    #     height = resolution[0]
    #     width = resolution[1]
    #     # TODO Load intrinsic
    #     intrinsics = []
    #     for idx in indices:
    #         # intrinsics.append(self.camera.get_adapted_intrinsic(resolution, idx))
    #         intrinsics.append(self.camera.load_intrinsics(idx))

    #     projected_points = np.zeros((len(indices), n_points, 2), dtype=int)
    #     visible_points_view = np.zeros((len(indices), n_points), dtype=bool)
    #     print(f"[INFO] Computing the visible points in each view.")

    #     for i, idx in tqdm(enumerate(indices), total=len(indices)):  # for each view
    #         # *******************************************************************************************************************
    #         # STEP 1: get the projected points
    #         # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
    #         projected_points_not_norm = (intrinsics[i] @ poses[i] @ X.T).T
    #         # Get the mask of the points which have a non-null third coordinate to avoid division by zero
    #         mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
    #         # Get non homogeneous coordinates of valid points (2D in the image)
    #         projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
    #                 projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
    #         # *******************************************************************************************************************
    #         # STEP 2: occlusions computation
    #         # Load the depth from the sensor
    #         depth_path = os.path.join(depths_path, str(idx) + '.png')
    #         sensor_depth = imageio.imread(depth_path) / depth_scale
    #         inside_mask = (projected_points[i,:,0] >= 25) * (projected_points[i,:,1] >= 25) \
    #                             * (projected_points[i,:,0] < width - 25) \
    #                             * (projected_points[i,:,1] < height - 25)
    #         pi = projected_points[i].T
    #         # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
    #         point_depth = projected_points_not_norm[:,2]
    #         # Compute the visibility mask, true for all the points which are visible from the i-th view
    #         visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
    #                                     - point_depth[inside_mask]) <= \
    #                                     vis_threshold).astype(bool)
    #         inside_mask[inside_mask == True] = visibility_mask
    #         visible_points_view[i] = inside_mask
    #     return visible_points_view, projected_points, resolution

    def get_visible_points_view(self): # Refactor the function to avoid the for loop and optimize the computation
        vis_threshold = self.vis_threshold
        indices = list(self.indices)
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)  # expected (V,4,4)
        X = self.point_cloud.get_homogeneous_coordinates()  # (N,4)
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path

        # resolution (H, W) - same as original
        resolution = imageio.imread(os.path.join(depths_path, f"{indices[0]}.png")).shape
        height, width = resolution[0], resolution[1]

        # load intrinsics list in same order as indices
        intrinsics = [self.camera.load_intrinsics(idx) for idx in indices]

        projected_points = np.zeros((len(indices), n_points, 2), dtype=np.int32)
        visible_points_view = np.zeros((len(indices), n_points), dtype=bool)

        print("[INFO] Computing the visible points in each view.")

        for i, idx in enumerate(indices):
            # STEP 1: project
            projected_points_not_norm = (intrinsics[i] @ poses[i] @ X.T).T  # (N,3)
            z = projected_points_not_norm[:, 2]  # (N,)

            # IMPORTANT: keep the original condition (z != 0) rather than abs(z)>eps
            valid_z = (z != 0)

            # Compute u,v without producing nan/inf for invalid points
            u = np.zeros(n_points, dtype=np.float32)
            v = np.zeros(n_points, dtype=np.float32)

            # Safe division only where valid_z
            np.divide(projected_points_not_norm[:, 0], z, out=u, where=valid_z)
            np.divide(projected_points_not_norm[:, 1], z, out=v, where=valid_z)

            # If division yields non-finite values (z tiny, or bad inputs), force them out-of-bounds
            # This avoids cast warnings AND preserves original behavior (these points should not become valid pixels).
            bad = valid_z & (~np.isfinite(u) | ~np.isfinite(v))
            if np.any(bad):
                u[bad] = -1e9
                v[bad] = -1e9

            ui = u.astype(np.int32)
            vi = v.astype(np.int32)

            projected_points[i, :, 0] = ui
            projected_points[i, :, 1] = vi

            # STEP 2: occlusions computation (same as original)
            depth_path = os.path.join(depths_path, str(idx) + ".png")
            sensor_depth = imageio.imread(depth_path) / depth_scale

            inside_mask = (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)

            # Depth from projection (same as original)
            point_depth = z

            if np.any(inside_mask):
                # pixel indices for inside points only
                xs = ui[inside_mask]
                ys = vi[inside_mask]

                visibility_mask = (np.abs(sensor_depth[ys, xs] - point_depth[inside_mask]) <= vis_threshold)
                # same effect as original: inside_mask[inside_mask==True] = visibility_mask
                tmp = inside_mask.copy()
                tmp[inside_mask] = visibility_mask
                visible_points_view[i] = tmp
            else:
                visible_points_view[i] = inside_mask  # all false

        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    # def get_visible_points_in_view_in_mask(self):
    #     masks = self.masks
    #     num_view = len(self.indices)
    #     visible_points_view, projected_points, resolution = self.get_visible_points_view()
    #     visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
    #     print(f"[INFO] Computing the visible points in each view in each mask.")
    #     for i in tqdm(range(num_view)):
    #         for j in range(masks.num_masks):
    #             visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
    #             proj_points = projected_points[i][visible_masks_points]
    #             if(len(proj_points) != 0):
    #                 visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
    #     self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
    #     self.visible_points_view = visible_points_view
    #     self.projected_points = projected_points
    #     self.resolution = resolution
    #     return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    def get_visible_points_in_view_in_mask(self): # Refactor the function to avoid the for loop and optimize the computation
        masks = self.masks
        num_view = len(self.indices)

        visible_points_view, projected_points, resolution = self.get_visible_points_view()
        H, W = resolution[0], resolution[1]

        # original expects masks.masks shape (N, M)
        mm = masks.masks.astype(bool)  # (N, M)
        num_masks = masks.num_masks

        visible_points_in_view_in_mask = np.zeros((num_view, num_masks, H, W), dtype=bool)

        print("[INFO] Computing the visible points in each view in each mask.")

        for i in range(num_view):
            vis_idx = np.flatnonzero(visible_points_view[i])
            if vis_idx.size == 0:
                continue

            pp = projected_points[i, vis_idx]  # (K,2)
            x = pp[:, 0]
            y = pp[:, 1]

            # membership of visible points across all masks: (K, M)
            mem = mm[vis_idx, :]  # bool

            # Fill all masks at once for this view.
            mask_ids, point_ids = np.nonzero(mem.T)
            if point_ids.size > 0:
                visible_points_in_view_in_mask[i, mask_ids, y[point_ids], x[point_ids]] = True

        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
        self.visible_points_view = visible_points_view
        self.projected_points = projected_points
        self.resolution = resolution
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution

    def get_visibility_mat(self, topk=15, return_topk_indices=False):
        # masks.masks: (N, M), visible_points_view: (V, N)
        pred_masks_3d = torch.from_numpy(self.masks.masks.T.astype(np.float32))
        inside_mask = torch.from_numpy(self.visible_points_view.astype(np.float32))

        # (M, N) x (V, N) -> (M, V)
        intersection = torch.einsum("ik,fk->if", pred_masks_3d, inside_mask)
        total_point_number = pred_masks_3d.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visibility_matrix = intersection / total_point_number

        topk = min(topk, visibility_matrix.shape[-1])
        if topk <= 0:
            empty_idx = torch.empty((visibility_matrix.shape[0], 0), dtype=torch.long)
            empty_bool = torch.zeros_like(visibility_matrix, dtype=torch.bool)
            if return_topk_indices:
                return empty_bool, empty_idx
            return empty_bool

        max_visibility_in_frame = torch.topk(visibility_matrix, topk, dim=-1).indices

        visibility_matrix_bool = torch.zeros_like(visibility_matrix, dtype=torch.bool)
        row_idx = torch.arange(visibility_matrix_bool.shape[0], dtype=torch.long)[:, None]
        visibility_matrix_bool[row_idx, max_visibility_in_frame] = True
        if return_topk_indices:
            return visibility_matrix_bool, max_visibility_in_frame
        return visibility_matrix_bool
    
    def get_top_k_indices_per_mask(self, k):
        _, topk_indices_per_mask = self.get_visibility_mat(topk=k, return_topk_indices=True)
        topk_indices_per_mask = topk_indices_per_mask.cpu().numpy()
        return topk_indices_per_mask
    # def get_top_k_indices_per_mask(self, k):
    #     V = len(self.indices)
    #     M = self.masks.num_masks
    #     vpm = self.visible_points_in_view_in_mask

    #     counts = np.zeros((V, M), dtype=np.int32)
    #     for v in range(V):
    #         row = vpm[v]
    #         for m in range(M):
    #             counts[v, m] = row[m][0].size  # x.size

    #     k_eff = min(k, V)
    #     # get k largest (unordered)
    #     idx_part = np.argpartition(-counts, kth=k_eff-1, axis=0)[:k_eff, :]  # (k_eff, M)

    #     # sort those k by actual counts descending
    #     part_counts = np.take_along_axis(counts, idx_part, axis=0)  # (k_eff, M)
    #     order = np.argsort(-part_counts, axis=0)                    # (k_eff, M)
    #     topk = np.take_along_axis(idx_part, order, axis=0).T        # (M, k_eff)
    #     return topk

    
class FeaturesExtractor:
    def __init__(self, 
                 camera, 
                 clip_model, 
                 images, 
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device):
        self.camera = camera
        self.images = images
        self.device = device
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        # self.sam_m, self.sam_p = init_sam_model(device)

        print("[INFO] Loading clip model")
        self.clip_type = "FG_CLIP"
        self.clip_model, self.clip_preprocess = init_clip_visual("qihoo360/fg-clip-large") #clip_model="google/siglip2-large-patch16-512")#"qihoo360/fg-clip-large")

        # if "fg-clip" in clip_model:
        #     self.clip_model = AutoModelForCausalLM.from_pretrained(clip_model, trust_remote_code=True).to(device)
        #     self.clip_preprocess = AutoImageProcessor.from_pretrained(clip_model).preprocess
        # else:
        #     self.clip_model, self.clip_preprocess = clip.load(clip_model, device)

    def merge_masks(self, min_points=100, outfolder=None):
        # Filter masks based on the number of points
        pred_masks = self.point_projector.masks.masks.T.astype(bool)
        print(f"Number of original masks: {pred_masks.shape[0]}")
        valid_mask_indices = np.where(np.sum(pred_masks, axis=1) > min_points)[0]  # Get indices of valid masks
        print(f"Number of valid masks after filtering: {len(valid_mask_indices)}")
        pred_masks = pred_masks[valid_mask_indices]  # Filter masks based on the valid indices
        # openmask3d_features = openmask3d_features[valid_mask_indices]  # Filter features based on the valid indices

        graph = construct_graph(pred_masks)
        clusters = list(nx.connected_components(graph))
        mask_index_mapping = []  # List of list of original indices
        for group in clusters:
            mask_index_mapping.append(list(group))
        print(f"Clusters found: {mask_index_mapping}")
        standalones = [i for i, degree in graph.degree() if degree == 0]
        print(f"Standalone masks: {standalones}")

        MERGE_POLICIES = ["majority_vote", "logical_or", "logical_and"]
        merge_policy = "majority_vote"  # Define the merging policy

        merged_masks = []
        merged_features = []
        for cluster in clusters:
            combined_mask = np.zeros_like(pred_masks[0], dtype=int)
            # combined_features = np.zeros_like(openmask3d_features[0], dtype=openmask3d_features.dtype)
            vote_threshold = (len(cluster) // 2) + 1  # Majority vote threshold
            for idx in cluster:
                # combined_mask |= pred_masks[idx]
                combined_mask += pred_masks[idx].astype(int)  # Use addition to merge masks
                # combined_features += openmask3d_features[idx]  # Sum the features for merging
            if merge_policy == "logical_or":
                combined_mask = (combined_mask > 0).astype(bool)
            elif merge_policy == "logical_and":
                combined_mask = (combined_mask == len(cluster)).astype(bool)
            elif merge_policy == "majority_vote":
                combined_mask = (combined_mask >= vote_threshold).astype(bool)  # Convert to boolean mask

            merged_masks.append(combined_mask)
            # merged_features.append(combined_features / len(cluster))  # Average the features

        merged_masks = np.array(merged_masks).astype(bool)
        # merged_features = np.array(merged_features)
        print(f"Number of merged masks: {len(merged_masks)}")

        # save the merged masks and features
        # save mask as pytrorch tensor
        torch.save(merged_masks.T, os.path.join(outfolder, "merged_masks.pt"))
        # np.save(os.path.join(outfolder, "merged_features.npy"), merged_features)
        self.point_projector.masks = InstanceMasks3D(masks_path=os.path.join(outfolder, "merged_masks.pt"))  # Update the masks in the point projector
    
    def clean_masks(self, eps=0.05, min_samples=5):
        """
        Clean the masks by removing outliers using DBSCAN.
        """
        points = np.asarray(self.point_projector.point_cloud.points)
        for mask_idx, original_mask in enumerate(self.point_projector.masks.masks.T):
            # print(f"{mask_idx}:")
            # get color from matplotlib colormap
            p = points[np.where(original_mask)[0]]
            c = None

            mask = original_mask.nonzero()[0]

            filtered_idx, _, _ = clean_point_cloud(p, c, eps=eps, min_samples=min_samples)

            mask = mask[filtered_idx]  # Update the mask with the filtered indices
            np_zeros = np.zeros_like(original_mask, dtype=bool)
            np_zeros[mask] = True  # Create a new mask with the filtered indices
            # update the mask
            # Remap the filtered indices to the original mask
            self.point_projector.masks.masks[:, mask_idx] = np_zeros

    def preprocess_images(self, images):
        if self.clip_type == "FG_CLIP":
            return self.clip_preprocess(images, return_tensors="pt")['pixel_values'].to(self.device) # (n, 512)
        else:
            return self.clip_preprocess(images).to(self.device) # (n, 768)
    
    def get_image_features(self, images):
        if self.clip_type == "FG_CLIP":
            image_pt = torch.cat(images)
            return self.clip_model.get_image_features(image_pt)
        else:
            image_pt = torch.stack(images)
            return self.clip_model.encode_image(image_pt.to(self.device)).float()

    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        if self.clip_type == "FG_CLIP":
            mask_clip = np.zeros((num_masks, self.clip_model.config.text_config.hidden_size)) #initialize mask clip
        else:
            mask_clip = np.zeros((num_masks, 768))

        np_images = self.images.get_as_np_list()
        mask_view_crop = {"bbox_format": "xyxy",
                          "meta": {
                              "num_masks": num_masks,
                              "num_views_per_mask": topk,
                              "image_size": np_images[0].shape[:2]
                            },
                            "masks": {}
                          }  # Dictionary to store the crops for each mask and view
        for mask in tqdm(range(num_masks)): # for each mask
            mask_view_crop['masks'][mask] = {}
            images_crops = []
            if(optimize_gpu_usage):
                self.clip_model.to(torch.device('cpu'))
                # self.predictor_sam.model.cuda()
                self.sam_m.to(torch.device('cuda'))
            for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view
                view_id = self.images.indices[view]
                # mask_view_crop['masks'][mask][view_id] = list()
                if(optimize_gpu_usage):
                    torch.cuda.empty_cache()
                
                # Get original mask points coordinates in 2d images
                point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                if (point_coords.shape[0] > 0):
                    mask_view_crop['masks'][mask][view_id] = list()
                    # start = time.time()
                    self.predictor_sam.set_image(np_images[view])
                    
                    # # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    # print(f"SAM inference time: {time.time() - start:.2f}s")

                    # start = time.time()
                    # best_mask = process_sam_points(
                    #     self.sam_m, self.sam_p, np_images[view], point_coords, num_random_rounds, num_selected_points
                    # )
                    # print(f"RSAM processing time: {time.time() - start:.2f}s")

                    # CROP SAM masked image
                    erosion_kernel = np.ones((3, 3), np.uint8)
                    erosed_mask = cv2.erode(best_mask.astype(np.uint8), erosion_kernel, iterations=1)
                    # dilation_kernel = np.ones((5, 5), np.uint8)
                    # dilated_mask = cv2.dilate(best_mask.astype(np.uint8), dilation_kernel, iterations=2).astype(bool)
                    # mask_img = np.array(self.images.images[view])
                    mask_img = np_images[view].copy()
                    try:
                        mask_img[np.where(erosed_mask == False)] = [255, 255, 255]
                        mask_img = Image.fromarray(mask_img)
                    except Exception as e:
                        print(f"Error processing mask image {view}: {e}")
                        self.images.images[view].save(os.path.join(out_folder, f"crop{mask}_{view_id}.png"))
                        continue
                    # cropped_img = mask_img.crop((x1, y1, x2, y2))

                    #TODO Decide one mechanism for visual embedding computation
                    images_crops.append(self.preprocess_images(self.images.images[view])) # add the full image as well
                    
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level+1, multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))                       
                        if(save_crops) and (level == num_levels - 1):
                            # Save the masked image
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view_id}_{level}.png"))
                        # I compute the CLIP feature using the standard clip model
                        # cropped_img_processed = self.clip_preprocess(cropped_img)
                        cropped_img_processed = self.preprocess_images(cropped_img)
                        images_crops.append(cropped_img_processed)
                        mask_view_crop['masks'][mask][view_id].append([x1, y1, x2, y2])

                        # if level == num_levels - 1: # save the last level cropped mask as well
                        #     # masked image
                        #     cropped_mask = mask_img.crop((x1, y1, x2, y2))
                        #     if(save_crops):
                        #         # save the masked image
                        #         cropped_mask.save(os.path.join(out_folder,  f"crop{mask}_{view_id}_{level+1}.png"))
                        #     # cropped_mask_processed = self.clip_preprocess(cropped_mask)
                        #     cropped_mask_processed = self.preprocess_images(cropped_mask) # return tensors for the last level
                        #     images_crops.append(cropped_mask_processed)
                        
                        cropped_mask = mask_img.crop((x1, y1, x2, y2))
                        if(save_crops) and (level == num_levels - 1):
                            cropped_mask.save(os.path.join(out_folder, f"crop{mask}_{view_id}_{level}_.png"))
                        # cropped_mask_processed = self.clip_preprocess(cropped_mask)
                        cropped_mask_processed = self.preprocess_images(cropped_mask) # return tensors for the last level
                        images_crops.append(cropped_mask_processed)
            
            if(optimize_gpu_usage):
                # self.predictor_sam.model.cpu()
                self.sam_m.to(torch.device('cpu'))
                self.clip_model.to(torch.device('cuda'))                
            if(len(images_crops) > 0):
                # image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    # image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    # image_features = self.clip_model.get_image_features(image_input)
                    image_features = self.get_image_features(images_crops)
                    image_features /= image_features.norm(dim=-1, keepdim=True) #normalize

                # Scale the last feature to be its 3 times
                # TODO
                # image_features[3::4] = 3 * image_features[3::4]
                # image_features[0::3] = 4 * image_features[0::3]
                mask_clip[mask] = image_features.mean(axis=0).cpu().float().numpy()

        return mask_clip, mask_view_crop
