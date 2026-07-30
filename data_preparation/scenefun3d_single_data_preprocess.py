import sys
import os
import os.path

from t_funs3d.utils.sun3d.data_parser import DataParser
import open3d as o3d
import numpy as np
import cv2  # Needed for cv2.Rodrigues

# from the scenefun3d toolkit but converted 
def convert_angle_axis_to_matrix3(angle_axis):
    """
    Converts a rotation from angle-axis to a 3x3 rotation matrix.
    """
    if not isinstance(angle_axis, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if angle_axis.shape != (3,):
        raise ValueError("Input must be a 3-element array.")
    
    matrix, _ = cv2.Rodrigues(angle_axis)
    return matrix


def convert_traj_to_pose_folder(traj_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(traj_path, 'r') as f:
        lines = f.readlines()
    
    ts_path = os.path.join(os.path.dirname(output_folder), "timestamps.txt")
    seq_id = os.path.basename(os.path.dirname(os.path.dirname(output_folder)))
    print(seq_id)

    timestamps = []

    for _, line in enumerate(lines):
        tokens = line.strip().split()
        assert len(tokens) == 7
        ts = tokens[0]
        timestamps.append(seq_id + "_" + ts)

        ax, ay, az, tx, ty, tz = map(float, tokens[1:])
        angle_axis = np.array([ax, ay, az])
        translation = np.array([tx, ty, tz])

        rotation_matrix = convert_angle_axis_to_matrix3(angle_axis)

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation
        pose_matrix = np.linalg.inv(pose_matrix)  # Invert the pose matrix

        out_path = os.path.join(output_folder, f"{seq_id}_{ts}.txt")
        np.savetxt(out_path, pose_matrix, fmt="%.6f")

    # print(f"Saving timestamps to: {ts_path}")
    with open(ts_path, 'w') as ts_file:
        for ts in timestamps:
            ts_file.write(f"{ts}\n")

    print(f"Converted {len(lines)} poses to folder: {output_folder}")

def remove_top_surface_fast(pcd, voxel_size=0.01, ceiling_thickness=0.05):
    """
    Efficiently remove the top-most points (max z) for each (x,y) cell.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input cloud
        voxel_size (float): Grid size for grouping (meters)
        ceiling_thickness (float): Remove all points within this distance
                                   from the local max z in each cell
    
    Returns:
        pcd_filtered (o3d.geometry.PointCloud): Cloud without ceiling
    """
    points = np.asarray(pcd.points)

    # Discretize x,y into grid cells
    xy_keys = np.round(points[:, :2] / voxel_size).astype(np.int64)

    # Sort by x,y then z
    order = np.lexsort((points[:, 2], xy_keys[:, 1], xy_keys[:, 0]))
    xy_sorted = xy_keys[order]
    z_sorted = points[order, 2]

    # Find the max z per (x,y) group
    _, idx_start = np.unique(xy_sorted, axis=0, return_index=True)
    _, idx_end   = np.unique(xy_sorted, axis=0, return_index=True)

    # Take group boundaries
    idx_start = np.r_[idx_start, len(z_sorted)]
    keep_mask = np.ones(len(points), dtype=bool)

    # Iterate only unique cells (much fewer than total points)
    for i in range(len(idx_start) - 1):
        start, end = idx_start[i], idx_start[i+1]
        z_group = z_sorted[start:end]
        z_max = z_group.max()
        # Mark as False all points within thickness of max z
        drop = (z_group >= z_max - ceiling_thickness)
        keep_mask[order[start:end][drop]] = False

    # Apply mask
    pcd_filtered = pcd.select_by_index(np.where(keep_mask)[0])
    return pcd_filtered

def save_matrix_to_txt(matrix, out_path):
    """
    Saves a 4x4 matrix to a .txt file with space-separated values.
    """
    np.savetxt(out_path, matrix, fmt="%.6f")

def keep_largest_dbscan_cluster(pcd: o3d.geometry.PointCloud, 
                                eps=0.02, 
                                min_points=10,
                                keep_top_k=1) -> o3d.geometry.PointCloud:
    """
    Cluster with DBSCAN and keep the largest K clusters.
    """
    if (len(pcd.points) == 0):
        print("Warning: Empty point cloud provided to keep_largest_dbscan_cluster.")
        return pcd
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    if labels.size == 0:
        print("Warning: DBSCAN returned no clusters.")
        return pcd
    
    # Exclude noise points
    valid = labels >= 0
    if not np.any(valid):
        print("Warning: DBSCAN found only noise points.")
        return pcd
    
    # Count cluster sizes
    unique, counts = np.unique(labels[valid], return_counts=True)
    order = np.argsort(counts)[::-1]  # Descending by size
    chosen = set(unique[order[:keep_top_k]])  # Keep top K clusters

    idx = np.where(np.isin(labels, list(chosen)))[0]
    return pcd.select_by_index(idx)


def main(base_dir, visit_id, split):

    scene_dir = os.path.join(base_dir, split, visit_id)

    # Get all sequences in the scene directory
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    
    ################ Read and process the laser scan point cloud  #################
    dataParser = DataParser(base_dir, split)
    laser_pc = dataParser.get_laser_scan(visit_id)
    cropped_pc = dataParser.get_cropped_laser_scan(visit_id, laser_pc)

    print(f"Original: {len(laser_pc.points)} points, {len(laser_pc.colors)} colors")
    print(f"Cropped: {len(cropped_pc.points)} points, {len(cropped_pc.colors)} colors")
    # o3d.visualization.draw_geometries([cropped_pc])
    o3d.io.write_point_cloud(f"./{visit_id}_laser_scan_cleaned.ply", cropped_pc)

    z_vals = np.asarray(cropped_pc.points)[:, 2]
    print(f"Z range: min={z_vals.min():.2f}, max={z_vals.max():.2f}")

    cropped_pc = cropped_pc.voxel_down_sample(voxel_size=0.008)

    print("Removing ceiling")
    cropped_pc = remove_top_surface_fast(cropped_pc)
    # o3d.visualization.draw_geometries([cropped_pc])

    z_vals_no_ceiling = np.asarray(cropped_pc.points)[:, 2]
    print(f"Z range: min={z_vals_no_ceiling.min():.2f}, max={z_vals_no_ceiling.max():.2f}")
    print(f"Cropped: {len(cropped_pc.points)} points, {len(cropped_pc.colors)} colors")
    # no_ceiling_path = f"./{visit_id}_laser_scan_no_ceiling.ply"
    # o3d.io.write_point_cloud(no_ceiling_path, cropped_pc)

    downsampled_pc = keep_largest_dbscan_cluster(cropped_pc, eps=0.05, min_points=20, keep_top_k=1)
    print("Keeping largest DBSCAN cluster")
    print(f"Preprocessed point cloud has {len(downsampled_pc.points)} points")
    o3d.visualization.draw_geometries([downsampled_pc])
    
    # Save the preprocessed point cloud
    o3d.io.write_point_cloud(f"{scene_dir}/{visit_id}_laser_scan_downsampled.ply", downsampled_pc)


    ############## Convert poses ##############
    sequences = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]

    # Extract the ID from the sequence paths
    print(f"Available sequences in {scene_dir}: {sequences}")

    # Convert trajectory files to pose folders for each sequence
    for seq_id in sequences:
        data_dir = os.path.join(scene_dir, seq_id)

        # Check if the scene_id and visit_id are valid
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        prepose_dir = os.path.join(data_dir, "processed")

        # Convert trajectory file to pose folder
        traj_file = os.path.join(data_dir, "hires_poses.traj")
        output_dir = os.path.join(prepose_dir, "pose")
        convert_traj_to_pose_folder(traj_file, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process SceneFun3D data.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the SceneFun3D dataset.")
    parser.add_argument("--visit_id", type=str, required=True, help="Visit ID of the SceneFun3D dataset.")
    parser.add_argument("--split", type=str, required=True, help="Split of the SceneFun3D dataset.")
    args = parser.parse_args()

    main(args.base_dir, args.visit_id, args.split)
   