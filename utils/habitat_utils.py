import gzip
import os
import time

from PIL import Image
import cv2
from habitat.utils.visualizations import maps
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import jsonlines
from matplotlib import pyplot as plt
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R


def make_sim_setting_dict(scene, cam_config, action_config, seed=1):
    """Make sim setting dictionary."""
    sim_settings = {
        "width": cam_config.WIDTH,
        "height": cam_config.HEIGHT,
        "scene": scene,
        "sensor_height": cam_config.SENSOR_HEIGHT,
        "color_sensor": cam_config.RGB_SENSOR,
        "color_360_sensor": cam_config.RGB_360_SENSOR,
        "depth_sensor": cam_config.DEPTH_SENSOR,
        "semantic_sensor": cam_config.SEMANTIC_SENSOR,
        "seed": seed,
        "enable_physics": False,
        "forward_amount": action_config.FORWARD_AMOUNT,
        "backward_amount": action_config.BACKWARD_AMOUNT,
        "turn_left_amount": action_config.TURN_LEFT_AMOUNT,
        "turn_right_amount": action_config.TURN_RIGHT_AMOUNT,
    }

    return sim_settings


def make_cfg(settings):
    """Make config object with sim setting dictionary."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    sensor_specs = []

    if (settings["color_sensor"] is True) and (settings["color_360_sensor"] is True):
        raise ValueError("Normal camera & 360 camera cannot be used at the same time.")

    if settings["color_sensor"] is True:
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

    if settings["color_360_sensor"] is True:
        color_360_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        color_360_sensor_spec.uuid = "color_sensor"
        color_360_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_360_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_360_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_360_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(color_360_sensor_spec)

    if settings["depth_sensor"] is True:
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"] is True:
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings["forward_amount"])
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=settings["backward_amount"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["turn_left_amount"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["turn_right_amount"])
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_output_path(output_path, scene_number):
    """Make directory for output file & records."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    observation_path = os.path.join(output_path, f"observation_{scene_number}", timestr)
    os.makedirs(observation_path, exist_ok=True)
    pos_record_json = os.path.join(output_path, f"observation_{scene_number}", f"pos_record_{timestr}.json")

    return observation_path, pos_record_json


def print_scene_recur(scene, limit_output=10):
    """Print semantic information in scene file recursively."""
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(f"Level id:{level.id}, center:{level.aabb.center}," f" dims:{level.aabb.sizes}")
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return


def display_observation(rgb_obs, semantic_obs, depth_obs):
    """Display sensor observation image."""
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]
    titles = ["rgb", "semantic", "depth"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


def init_opencv_cam(x_size=1152, y_size=1152):
    """Create image window for observation."""
    cv2.namedWindow("observation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("observation", x_size, y_size)


def display_opencv_cam(rgb_obs) -> int:
    """Draw nodes and edges into map image."""
    cv2.imshow("observation", rgb_obs)
    key = cv2.waitKey()

    return key


def init_map_display(window_name="map", x_size=1152, y_size=1152):
    """Create image window for map."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, x_size, y_size)


def display_map(topdown_map, window_name="map", key_points=None, wait_for_key=False):
    """Display a topdown map with OpenCV."""

    if key_points is not None:
        for pnt in key_points:
            cv2.drawMarker(
                img=topdown_map,
                position=(int(pnt[1]), int(pnt[0])),
                color=(255, 0, 0),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=1,
            )

    if key_points is not None and len(key_points) >= 2:
        for i in range(len(key_points) - 1):
            cv2.line(
                img=topdown_map,
                pt1=(int(key_points[i][1]), int(key_points[i][0])),
                pt2=(int(key_points[i + 1][1]), int(key_points[i + 1][0])),
                color=(0, 255, 0),
                thickness=1,
            )

    cv2.imshow(window_name, topdown_map)
    if wait_for_key:
        cv2.waitKey()


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


def get_entire_maps_by_levels(sim, meters_per_pixel):
    """Sample random maps & Get the largest map by levels."""
    print("Sampling maps to get proper map...")
    nav_point_list = []
    closest_level_list = []
    for i in range(300):
        nav_point = sim.pathfinder.get_random_navigable_point()
        distance_list = []
        average_list = []
        for level in sim.semantic_scene.levels:
            for region in level.regions:
                distance = abs(region.aabb.center[1] - (nav_point[1] + 0.5))
                distance_list.append(distance)
            average = sum(distance_list) / len(distance_list)
            average_list.append(average)
        closest_level = average_list.index(min(average_list))
        nav_point_list.append(nav_point)
        closest_level_list.append(closest_level)
    print("Map sampling done.")

    print("Selecting proper maps on desired level")
    recolored_topdown_map_list = []
    topdown_map_list = []
    height_points = []
    for level_id in range(len(sim.semantic_scene.levels)):
        area_size_list = []
        for i, point in enumerate(nav_point_list):
            area_size = 0

            if not sim.pathfinder.is_navigable(point):
                print("Sampled point is not navigable")

            if closest_level_list[i] == level_id:
                topdown_map = maps.get_topdown_map(sim.pathfinder, height=point[1], meters_per_pixel=meters_per_pixel)
                area_size = np.count_nonzero(topdown_map == 1)

            area_size_list.append(area_size)

        sample_id = area_size_list.index(max(area_size_list))
        topdown_map = maps.get_topdown_map(
            sim.pathfinder, height=nav_point_list[sample_id][1], meters_per_pixel=meters_per_pixel
        )
        height_points.append(nav_point_list[sample_id][1])
        topdown_map_list.append(topdown_map)
        recolor_palette = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        recolored_topdown_map = recolor_palette[topdown_map]
        recolored_topdown_map_list.append(recolored_topdown_map)
    print("Map selection done.")

    return recolored_topdown_map_list, topdown_map_list, height_points


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
