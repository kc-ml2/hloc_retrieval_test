import argparse
import json
import os
import random

import cv2
import numpy as np

from global_localization.sim import HabitatSimWithMap
from utils.config_import import load_config_module
from utils.habitat_utils import draw_point_from_node, open_env_related_files
from utils.skeletonize_utils import topdown_map_to_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/concat_fourview_69FOV_HD.py")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--not-generate-test-query", action="store_true")
    parser.add_argument("--map-debug", action="store_true")
    parser.add_argument("--prune", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    not_generate_test_query = args.not_generate_test_query
    map_debug = args.map_debug
    pruning = args.prune

    config = load_config_module(module_name)
    output_path = config.PathConfig.LOCALIZATION_TEST_PATH

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, config, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            # Build binary top-down map & skeleton graph
            topdown_map = sim.topdown_map_list[level]
            graph = topdown_map_to_graph(topdown_map, config.DataConfig.REMOVE_ISOLATED, pruning)

            if len(list(graph.nodes)) == 0:
                continue

            # When debug mode, only run the codes below
            if map_debug:
                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
                for node_id in graph.nodes():
                    draw_point_from_node(map_image, graph, node_id)

                cv2.namedWindow("map", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("map", 1152, 1152)
                cv2.imshow("map", map_image)
                cv2.waitKey()
                continue

            # Make directory to save observation
            image_dir_by_scene = os.path.join(output_path, f"observation_{scene_number}")
            map_obs_result_path = os.path.join(image_dir_by_scene, f"{config.PathConfig.MAP_DIR_PREFIX}_{level}")
            os.makedirs(map_obs_result_path, exist_ok=True)

            # Save observation at every node
            for node_id in graph.nodes():
                _, _ = sim.set_state_from_grid(graph.nodes[node_id]["o"], level)
                observations = sim.get_cam_observations()
                color_img = observations["all_view"]

                if sim.multi_observations_in_single_node:
                    for i in range(config.CamConfig.NUM_CAMERA):
                        cv2.imwrite(
                            map_obs_result_path + os.sep + f"{node_id:06d}_{i}.jpg",
                            color_img[:, i * config.CamConfig.WIDTH : (i + 1) * config.CamConfig.WIDTH, :],
                        )
                else:
                    cv2.imwrite(map_obs_result_path + os.sep + f"{node_id:06d}.jpg", color_img)

            if not_generate_test_query:
                continue

            # Generate random query observation for test
            query_dirname = f"{config.PathConfig.QUERY_DIR_PREFIX}_{level}"
            test_query_path = os.path.join(image_dir_by_scene, query_dirname)
            os.makedirs(test_query_path, exist_ok=True)

            pos_record_json = os.path.join(
                image_dir_by_scene, f"{config.PathConfig.POS_RECORD_FILE_PREFIX}_{query_dirname}.json"
            )
            pos_record = {}
            pos_record.update({"scene_number": scene_number})
            pos_record.update({"level": level})

            # Sample only from explorable area, not outside the wall, not at the wall
            explorable_area_index = list(zip(*np.where(topdown_map == 1)))

            for k in range(config.DataConfig.NUM_TEST_SAMPLE_PER_LEVEL):
                grid_pos = random.sample(explorable_area_index, 1)[0]
                sim_pos, random_rotation = sim.set_state_from_grid(grid_pos, level)

                observations = sim.get_cam_observations()
                color_img = observations[sim.inference_view_attr]

                cv2.imwrite(test_query_path + os.sep + f"{k:06d}.jpg", color_img)

                record_sim_pos = {f"{k:06d}_sim": [[float(pos) for pos in sim_pos], random_rotation]}
                record_grid_pos = {f"{k:06d}_grid": [int(grid_pos[0]), int(grid_pos[1])]}
                pos_record.update(record_sim_pos)
                pos_record.update(record_grid_pos)

            with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                json.dump(pos_record, record_json, indent=4)

        sim.close()
