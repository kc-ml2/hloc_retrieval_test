import gzip
import os

import jsonlines
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R


def extrinsic_mat_list_to_pos_angle_list(ext_trans_mat_list):
    """Convert RxR dataset extrinsic matrix to Cartesian position & angle(quaternion) list."""
    pos_trajectory = []
    angle_trajectory = []

    for trans_mat in ext_trans_mat_list:
        position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
        pos_trajectory.append(position)
        angle_trajectory.append(angle_quaternion)

    return pos_trajectory, angle_trajectory


def interpolate_discrete_matrix(extrinsic_mat_list, interpolation_interval, translation_threshold):
    """Interpolate between two remote translation points."""
    for i, mat in enumerate(extrinsic_mat_list):
        if i + 1 < len(extrinsic_mat_list):
            next_mat = extrinsic_mat_list[i + 1]

        position = mat[:-1, 3]
        next_position = next_mat[:-1, 3]
        dist = np.sqrt(np.sum((next_position - position) ** 2, axis=0))
        num_interpolation = int(dist // interpolation_interval)

        interval_pos_list = []
        if dist > translation_threshold:
            interval_pos_list = np.linspace(position, next_position, num=num_interpolation)
            for n, interval_pos in enumerate(interval_pos_list):
                interval_mat = np.copy(mat)
                interval_mat[:-1, 3] = interval_pos
                extrinsic_mat_list.insert(i + n + 1, interval_mat)  # angle path is not interpolated

    return extrinsic_mat_list


def interpolate_discrete_path(pos_trajectory, angle_trajectory, interpolation_interval, translation_threshold):
    """Interpolate between two remote translation points."""
    for i, position in enumerate(pos_trajectory):
        if i + 1 < len(pos_trajectory):
            next_position = pos_trajectory[i + 1]

        dist = np.sqrt(np.sum((next_position - position) ** 2, axis=0))
        num_interpolation = int(dist // interpolation_interval)

        interval_pos_list = []
        if dist > translation_threshold:
            interval_pos_list = np.linspace(position, next_position, num=num_interpolation)
            for n, interval_pos in enumerate(interval_pos_list):
                pos_trajectory.insert(i + n + 1, interval_pos)
                angle_trajectory.insert(i + n + 1, angle_trajectory[i])  # angle path is not interpolated

    return pos_trajectory, angle_trajectory


def cal_pose_diff(trans_mat, prev_trans_mat):
    """Print position & angle diff."""
    inverse_mat = cal_inverse_transform_mat(prev_trans_mat)
    projected_mat = np.matmul(inverse_mat, trans_mat)
    position_diff, quaternion_diff = convert_transmat_to_point_quaternion(projected_mat)
    # Due to gimbal lock, conversion from quaternion to euler angle does not work properly.
    # So we used scipy Rotation to get euler angle from rotation matirix.
    rotation_mat = R.from_matrix(projected_mat[:3, :3])
    rotation_diff = rotation_mat.as_euler("zyx", degrees=True)

    return position_diff, quaternion_diff, rotation_diff


def cal_inverse_transform_mat(trans_mat):
    """Calculate inverse matirix of transformation matrix. It only works for transformation matrix."""
    inverse_mat = np.zeros([4, 4])
    inverse_mat[:3, :3] = trans_mat[:3, :3].transpose()
    inverse_mat[:-1, 3] = (-1) * np.matmul(trans_mat[:3, :3].transpose(), trans_mat[:-1, 3])
    inverse_mat[3, 3] = 1.0
    return inverse_mat


def remove_duplicate_matrix(extrinsic_mat_list):
    """Remove duplicate records to exculde stop motion."""
    deduplicated_mat_list = []
    for i, ext_trans_mat in enumerate(extrinsic_mat_list):
        if i == 0:
            prev_trans_mat = ext_trans_mat
        position_diff, _, rotation_diff = cal_pose_diff(ext_trans_mat, prev_trans_mat)
        prev_trans_mat = ext_trans_mat

        if (position_diff == [0.0, 0.0, 0.0]).all() and (rotation_diff == [0.0, 0.0, 0.0]).all():
            continue

        deduplicated_mat_list.append(ext_trans_mat)

    return deduplicated_mat_list


def convert_points_to_topdown(pathfinder, points, meters_per_pix):
    """Convert 3d points to 2d topdown coordinates."""
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for pnt in points:
        # convert 3D x,z to topdown x,y
        px = (pnt[0] - bounds[0][0]) / meters_per_pix
        py = (pnt[2] - bounds[0][2]) / meters_per_pix
        points_topdown.append(np.array([px, py]))
    return points_topdown


def convert_transmat_to_point_quaternion(trans_mat: np.ndarray):
    """Convert transformation matrix into position & quaternion."""
    position = trans_mat[:, -1][0:3]
    angle_quaternion = quaternion.from_rotation_matrix(trans_mat[0:3, 0:3])

    return position, angle_quaternion


def get_scene_by_eng_guide(instruction_id, train_guide_file, scene_directory):
    """Find & return scene glb file according to id of instruction."""
    jsonl_file = gzip.open(train_guide_file)
    reader = jsonlines.Reader(jsonl_file)

    print("Finding out scene according to instruction id...")
    for obj in reader:
        if obj["instruction_id"] == instruction_id and (obj["language"] == "en-IN" or obj["language"] == "en-US"):
            scene_number = obj["scan"]
            scene = scene_directory + os.sep + scene_number + os.sep + scene_number + ".glb"
            print("Found the scene.")

    if not scene:
        print("No scene found or the instruction is not in English.")

    return scene
