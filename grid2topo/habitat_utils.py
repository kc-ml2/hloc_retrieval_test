from PIL import Image
from typing import Tuple, Dict

import cv2
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from matplotlib import pyplot as plt
import numpy as np
import quaternion


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    # depth_sensor_spec = habitat_sim.CameraSensorSpec()
    # depth_sensor_spec.uuid = "depth_sensor"
    # depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    # depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    # depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    # depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # sensor_specs.append(depth_sensor_spec)

    # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    # semantic_sensor_spec.uuid = "semantic_sensor"
    # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    # semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    # semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "move_backward": habitat_sim.agent.ActionSpec("move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def print_scene_recur(scene, limit_output=10):
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
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


def display_opencv_cam(rgb_obs) -> int:
    """Draw nodes and edges into map image."""
    cv2.namedWindow("observation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("observation", 1000, 1000)
    cv2.imshow("observation", rgb_obs)
    key = cv2.waitKey()
    cv2.destroyAllWindows()

    return key


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


def display_map(topdown_map, key_points=None):
    """Display a topdown map with OpenCV."""

    if key_points is not None:
        for pnt in key_points:
            cv2.drawMarker(
                img=topdown_map,
                position=(int(pnt[0]), int(pnt[1])),
                color=(0, 255, 0),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=1,
            )

    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("map", 1000, 1000)
    cv2.imshow("map", topdown_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


def convert_transmat_to_point_quaternion(trans_mat: np.ndarray):
    """Convert transformation matrix into position & quaternion."""
    position = trans_mat[:,-1][0:3]
    angle_quaternion = quaternion.from_rotation_matrix(trans_mat[0:3,0:3])

    return position, angle_quaternion


def position_to_grid(position, img_shape, bounds):
    min_bound = bounds[0]
    max_bound = bounds[1]

    grid_ratio = img_shape[0] / (max_bound[2] - min_bound[2])
    grid_x = 0
    grid_y = 0
    return grid_x, grid_y


def static_to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    bounds: Dict[str, Tuple[float, float]],
) -> Tuple[int, int]:
    """Return gridworld index of realworld coordinates assuming top-left
    corner is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max). Same as the habitat-Lab maps.to_grid
    function but with a static `bounds` instead of requiring a simulator or
    pathfinder instance.
    """
    grid_size = (
        abs(bounds["upper"][2] - bounds["lower"][2]) / grid_resolution[0],
        abs(bounds["upper"][0] - bounds["lower"][0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - bounds["lower"][2]) / grid_size[0])
    grid_y = int((realworld_y - bounds["lower"][0]) / grid_size[1])
    return grid_x, grid_y