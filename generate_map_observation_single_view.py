import argparse
import json
import os
import random

import cv2
import numpy as np

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, DataConfig, PathConfig
from relocalization.sim import HabitatSimWithMap
from utils.habitat_utils import draw_point_from_node, open_env_related_files
from utils.skeletonize_utils import topdown_map_to_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--output-path", default="./single_view_output")
    parser.add_argument("--not-generate-test-sample", action="store_true")
    parser.add_argument("--map-debug", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    output_path = args.output_path
    height_json_path = args.map_height_json
    not_generate_test_sample = args.not_generate_test_sample
    map_debug = args.map_debug

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            # Build binary top-down map & skeleton graph
            topdown_map = sim.topdown_map_list[level]
            graph = topdown_map_to_graph(topdown_map, DataConfig.REMOVE_ISOLATED)

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
            observation_path = os.path.join(output_path, f"observation_{scene_number}")
            map_obs_result_path = os.path.join(observation_path, f"map_node_observation_level_{level}")
            os.makedirs(map_obs_result_path, exist_ok=True)

            # Save observation at every node
            for node_id in graph.nodes():
                _, _ = sim.set_state_from_grid(graph.nodes[node_id]["o"], level)
                observations = sim.get_cam_observations()
                color_img = observations["all_view"]

                for i in range(4):
                    cv2.imwrite(
                        map_obs_result_path + os.sep + f"{node_id:06d}_{i}.jpg",
                        color_img[:, i * 256 : (i + 1) * 256, :],
                    )

            if not_generate_test_sample:
                continue

            # Generate random sample observation for test
            test_sample_path = os.path.join(observation_path, f"test_sample_{level}")
            os.makedirs(test_sample_path, exist_ok=True)

            pos_record_json = os.path.join(observation_path, f"pos_record_test_sample_{level}.json")
            pos_record = {}
            pos_record.update({"scene_number": scene_number})
            pos_record.update({"level": level})

            # Sample only from explorable area, not outside the wall, not at the wall
            explorable_area_index = list(zip(*np.where(topdown_map == 1)))

            for k in range(TestConstant.NUM_SAMPLING_PER_LEVEL):
                grid_pos = random.sample(explorable_area_index, 1)[0]
                sim_pos, random_rotation = sim.set_state_from_grid(grid_pos, level)

                observations = sim.get_cam_observations()
                color_img = observations["all_view"][:, :256, :]

                cv2.imwrite(test_sample_path + os.sep + f"{k:06d}.jpg", color_img)

                record_sim_pos = {f"{k:06d}_sim": [[float(pos) for pos in sim_pos], random_rotation]}
                record_grid_pos = {f"{k:06d}_grid": [int(grid_pos[0]), int(grid_pos[1])]}
                pos_record.update(record_sim_pos)
                pos_record.update(record_grid_pos)

            with open(pos_record_json, "w") as record_json:  # pylint: disable=unspecified-encoding
                json.dump(pos_record, record_json, indent=4)

        sim.close()
