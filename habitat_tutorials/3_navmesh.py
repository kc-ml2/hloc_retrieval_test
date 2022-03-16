import argparse
import math
import os
import random

# function to display the topdown map
from PIL import Image
from habitat.utils.visualizations import maps
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils.common import d3_40_colors_rgb
import imageio
import magnum as mn
from matplotlib import pyplot as plt
import numpy as np


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

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
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(30, 15))

    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show(block=False)


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pix):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for pnt in points:
        # convert 3D x,z to topdown x,y
        px = (pnt[0] - bounds[0][0]) / meters_per_pix
        py = (pnt[2] - bounds[0][2]) / meters_per_pix
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for pnt in key_points:
            plt.plot(pnt[0], pnt[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)


output_path = "habitat_tutorials/output/"

if not os.path.exists(output_path):
    os.mkdir(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args, _ = parser.parse_known_args()
    test_scene = args.scene

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": test_scene,  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # the randomness is needed when choosing the actions
    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-3.0, 0.5, 0.0])  # world space
    agent_state.position = np.array([0.0, 0.5, 0.0])  # world space
    # agent_state.position = np.array([-1.8, 0.11, 19.33])  # world space
    agent.set_state(agent_state)

    # @markdown ###Configure Example Parameters:
    # @markdown Configure the map resolution:
    meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
    # @markdown ---
    # @markdown Customize the map slice height (global y coordinate):
    custom_height = True  # @param {type:"boolean"}
    height = 0.1  # @param {type:"slider", min:-10, max:10, step:0.1}
    # @markdown If not using custom height, default to scene lower limit.
    # @markdown (Cell output provides scene height range from bounding box for reference.)

    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    if not custom_height:
        # get bounding box minumum elevation for automatic height
        height = sim.pathfinder.get_bounds()[0][1]

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
        # This map is a 2D boolean array
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        print(np.shape(sim_topdown_map))
        input()

        # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module]
        # https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
        hablab_topdown_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel)
        print(np.shape(hablab_topdown_map))
        input()
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        print("Displaying the raw map from get_topdown_view:")
        display_map(sim_topdown_map)
        input()
        print("Displaying the map from the Habitat-Lab maps module:")
        display_map(hablab_topdown_map)
        input()

        # easily save a map to file:
        map_filename = os.path.join(output_path, "top_down_map.png")
        imageio.imsave(map_filename, hablab_topdown_map)

    ###################################################################################################################
    # @markdown ## Querying the NavMesh

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
        print("NavMesh area = " + str(sim.pathfinder.navigable_area))
        print("Bounds = " + str(sim.pathfinder.get_bounds()))

        # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
        pathfinder_seed = 4  # @param {type:"integer"}
        sim.pathfinder.seed(pathfinder_seed)
        nav_point = sim.pathfinder.get_random_navigable_point()
        print("Nav point: ", nav_point)
        input()
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

        print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

        # @markdown The closest boundary point can also be queried (within some radius).
        max_search_radius = 10.0  # @param {type:"number"}
        print("Distance to obstacle: " + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius)))
        hit_record = sim.pathfinder.closest_obstacle_surface_point(nav_point, max_search_radius)
        print("Closest obstacle HitRecord:")
        print(" point: " + str(hit_record.hit_pos))
        print(" normal: " + str(hit_record.hit_normal))
        print(" distance: " + str(hit_record.hit_dist))

        vis_points = [nav_point]
        print("Vis point: ", vis_points)
        input()

        # HitRecord will have infinite distance if no valid point was found:
        if math.isinf(hit_record.hit_dist):
            print("No obstacle found within search radius.")
        else:
            # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
            perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
            print("Perturbed point : " + str(perturbed_point))
            print("Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point)))
            snapped_point = sim.pathfinder.snap_point(perturbed_point)
            print("Snapped point : " + str(snapped_point))
            print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
            vis_points.append(snapped_point)

        print("Nav point: ", vis_points)
        input()

        # @markdown ---
        # @markdown ### Visualization
        # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
        meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

        xy_vis_points = convert_points_to_topdown(sim.pathfinder, vis_points, meters_per_pixel)
        print("xy vis point: ", xy_vis_points)
        input()
        # use the y coordinate of the sampled nav_point for the map height slice
        top_down_map = maps.get_topdown_map(sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        top_down_map = recolor_map[top_down_map]
        print("\nDisplay the map with key_point overlay:")
        display_map(top_down_map, key_points=xy_vis_points)
        input()

    ###################################################################################################################
    # @markdown ## Pathfinding Queries on NavMesh

    # @markdown The shortest path between valid points on the NavMesh can be queried as shown in this example.

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        seed = 1  # @param {type:"integer"}
        sim.pathfinder.seed(seed)

        # fmt off
        # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
        # fmt on
        sample1 = sim.pathfinder.get_random_navigable_point()
        sample2 = sim.pathfinder.get_random_navigable_point()

        # @markdown 2. Use ShortestPath module to compute path between samples.
        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points
        # @markdown - Success, geodesic path length, and 3D points can be queried.
        print("found_path : " + str(found_path))
        print("geodesic_distance : " + str(geodesic_distance))
        print("path_points : " + str(path_points))

        # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min

            top_down_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel)
            recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
            top_down_map = recolor_map[top_down_map]
            grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
            # convert world trajectory points to maps module grid points
            trajectory = [
                maps.to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )
                for path_point in path_points
            ]
            grid_tangent = mn.Vector2(trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0])
            path_initial_tangent = grid_tangent / grid_tangent.length()
            initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
            # draw the agent and trajectory on the map
            maps.draw_path(top_down_map, trajectory)
            maps.draw_agent(top_down_map, trajectory[0], initial_angle, agent_radius_px=8)
            print("\nDisplay the map with agent and path overlay:")
            display_map(top_down_map)
            input()

            # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
            display_path_agent_renders = True  # @param{type:"boolean"}
            if display_path_agent_renders:
                print("Rendering observations at path points:")
                tangent = path_points[1] - path_points[0]
                agent_state = habitat_sim.AgentState()
                for ix, point in enumerate(path_points):
                    if ix < len(path_points) - 1:
                        tangent = path_points[ix + 1] - point
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(point, point + tangent, np.array([0, 1.0, 0]))
                        tangent_orientation_q = mn.Quaternion.from_matrix(tangent_orientation_matrix.rotation())
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        agent.set_state(agent_state)

                        observations = sim.get_sensor_observations()
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]

                        display_sample(rgb, semantic, depth)
                        input()
