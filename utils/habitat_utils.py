"""
Original code:
https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb

The original code is released under the MIT license.

Modified by KC-ML2.
"""


import json
import os
import time

from PIL import Image
import cv2
from habitat.utils.visualizations import maps
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from matplotlib import pyplot as plt
import numpy as np


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
        color_sensor_spec.hfov = 69.0
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


def open_env_related_files(scene_list_file, height_json_path, scene_index=None):
    """Open cached file related to habitat sim."""
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    if scene_index is not None:
        if scene_index >= len(scene_list):
            raise IndexError(f"Scene list index out of range. The range is from 0 to {len(scene_list) - 1}")
        scene_list = [scene_list[scene_index]]

    return scene_list, height_data


def make_output_path(output_path, scene_number, file_prefix):
    """Make directory for output file & records."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    observation_path = os.path.join(output_path, f"observation_{scene_number}", timestr)
    os.makedirs(observation_path, exist_ok=True)
    pos_record_json = os.path.join(output_path, f"observation_{scene_number}", f"{file_prefix}_{timestr}.json")

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


def get_entire_maps_by_levels(sim, meters_per_pixel, num_random_samples=300):
    """Sample random maps & Get the largest map by levels."""
    print("Sampling maps to get proper map...")
    nav_point_list = []
    closest_level_list = []
    for i in range(num_random_samples):
        nav_point = sim.pathfinder.get_random_navigable_point()  # Get one random navigable point
        distance_list = []
        average_list = []

        # Iterate levels provided by Habitat dataset
        for level in sim.semantic_scene.levels:
            # Measure distance between current point and every spaces(rooms) in current level
            for region in level.regions:
                distance = abs(region.aabb.center[1] - (nav_point[1] + 0.5))
                distance_list.append(distance)

            average = sum(distance_list) / len(distance_list)
            average_list.append(average)  # Add average distance between this sample point and every room at this level

        closest_level = average_list.index(min(average_list))  # Estimate current level of the sample point
        # Store current point and estimated level. This process repeats with 300 samples
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

            # Get topdown map of all points that is estimated to be in this level
            if closest_level_list[i] == level_id:
                topdown_map = maps.get_topdown_map(sim.pathfinder, height=point[1], meters_per_pixel=meters_per_pixel)
                area_size = np.count_nonzero(topdown_map == 1)

            area_size_list.append(area_size)

        # Get the sample point and map with maximum area
        sample_id = area_size_list.index(max(area_size_list))
        topdown_map = maps.get_topdown_map(
            sim.pathfinder, height=nav_point_list[sample_id][1], meters_per_pixel=meters_per_pixel
        )
        height_points.append(nav_point_list[sample_id][1])
        topdown_map_list.append(topdown_map)

        # Recolor the map
        recolor_palette = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        recolored_topdown_map = recolor_palette[topdown_map]
        recolored_topdown_map_list.append(recolored_topdown_map)
    print("Map selection done.")

    return recolored_topdown_map_list, topdown_map_list, height_points


def draw_point_from_grid_pos(map_img, grid_pos, color):
    """Draw highlighted point(circle) on map with pixer position."""
    cv2.circle(
        img=map_img,
        center=(int(grid_pos[1]), int(grid_pos[0])),
        radius=2,
        color=color,
        thickness=-1,
    )


def draw_point_from_node(map_img, graph, node_id):
    """Draw point(circle) on map with negative radius."""
    cv2.circle(
        img=map_img,
        center=(int(graph.nodes()[node_id]["o"][1]), int(graph.nodes()[node_id]["o"][0])),
        radius=0,
        color=(0, 0, 255),
        thickness=-1,
    )


def highlight_point_from_node(map_img, graph, node_id, color):
    """Draw highlighted point(circle) on map with 1 radius."""
    cv2.circle(
        img=map_img,
        center=(int(graph.nodes()[node_id]["o"][1]), int(graph.nodes()[node_id]["o"][0])),
        radius=2,
        color=color,
        thickness=-1,
    )


def draw_line_from_edge(map_img, graph, edge_tuple, brightness):
    """Draw line with edge id tuple."""
    cv2.line(
        img=map_img,
        pt1=(int(graph.nodes[edge_tuple[0]]["o"][1]), int(graph.nodes[edge_tuple[0]]["o"][0])),
        pt2=(int(graph.nodes[edge_tuple[1]]["o"][1]), int(graph.nodes[edge_tuple[1]]["o"][0])),
        color=(int(255 * brightness), 122, 0),
        thickness=1,
    )


def save_observation(color_img, observation_path, img_id, pos_record, position, node_point):
    """Save current observation (RGB camera image) at designated directory & Update position record."""
    cv2.imwrite(observation_path + os.sep + f"{img_id:06d}.jpg", color_img)
    sim_pos = {f"{img_id:06d}_sim": [float(pos) for pos in position]}
    grid_pos = {f"{img_id:06d}_grid": [int(pnt) for pnt in node_point]}
    print("save image")

    # Update position record
    pos_record.update(sim_pos)
    pos_record.update(grid_pos)
