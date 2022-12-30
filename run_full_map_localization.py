import argparse
import json
import os

import cv2
import numpy as np
import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from config.algorithm_config import NetworkConstant, TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, DataConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import draw_point_from_node, highlight_point_from_node, open_env_related_files
from utils.skeletonize_utils import topdown_map_to_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    parser.add_argument("--load-model", default="./model_weights/model.20221129-125905.32batch.4view.weights.best.hdf5")
    parser.add_argument("--save-img", action="store_true")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path
    loaded_model = args.load_model
    save_img = args.save_img

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    # Load pre-trained model & top network
    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        siamese = ResnetBuilder.build_siamese_resnet_18
        model = siamese((NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS))
        model.load_weights(loaded_model, by_name=True)
        top_network = ResnetBuilder.build_top_network(model)

    # Main loop
    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            # Build binary top-down map & skeleton graph
            graph = topdown_map_to_graph(sim.topdown_map_list[level], DataConfig.REMOVE_ISOLATED)

            # Set file path
            map_cache_index = f"map_node_observation_level_{level}"
            sample_cache_index = f"test_sample_{level}"
            map_obs_dir = os.path.join(observation_path, map_cache_index)
            sample_dir = os.path.join(observation_path, sample_cache_index)
            map_embedding_file = os.path.join(observation_path, f"siamese_embedding_{map_cache_index}.npy")
            sample_embedding_file = os.path.join(observation_path, f"siamese_embedding_{sample_cache_index}.npy")
            sample_pos_record_file = os.path.join(observation_path, f"pos_record_test_sample_{level}.json")

            if args.save_img:
                visualization_result_path = os.path.join(observation_path, f"localization_visualize_result_{level}")
                os.makedirs(visualization_result_path, exist_ok=True)

            # Open files
            with open(map_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
                map_embedding_mat = np.load(f)
            with open(sample_embedding_file, "rb") as f:  # pylint: disable=unspecified-encoding
                sample_embedding_mat = np.load(f)
            with open(sample_pos_record_file, "r") as f:  # pylint: disable=unspecified-encoding
                sample_pos_record = json.load(f)

            num_map_embedding = np.shape(map_embedding_mat)[0]
            map_embedding_dimension = np.shape(map_embedding_mat)[1]
            sample_embedding_dimension = np.shape(sample_embedding_mat)[1]
            input_embedding_mat = np.zeros((num_map_embedding, map_embedding_dimension + sample_embedding_dimension))
            input_embedding_mat[:, :map_embedding_dimension] = map_embedding_mat

            for i, sample_embedding in enumerate(sample_embedding_mat):
                print("Sample No.: ", i)

                map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)

                for node in graph.nodes():
                    draw_point_from_node(map_image, graph, node)

                grid_pos = sample_pos_record[f"{i:06d}_grid"]
                sim_pos = sample_pos_record[f"{i:06d}_sim"]
                graph.add_node("current")
                graph.nodes()["current"]["o"] = grid_pos

                for input_embedding in input_embedding_mat:
                    input_embedding[map_embedding_dimension:] = sample_embedding

                with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
                    predictions = top_network.predict_on_batch(input_embedding_mat)

                similarity = predictions[:, 1]
                map_node_with_max_value = np.argmax(similarity)

                print("Max value: ", similarity[map_node_with_max_value], "   Node: ", map_node_with_max_value)

                high_similarity_set = similarity > TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
                for idx, upper in enumerate(high_similarity_set):
                    if upper:
                        highlight_point_from_node(map_image, graph, idx, (0, 0, 122))

                highlight_point_from_node(map_image, graph, map_node_with_max_value, (255, 255, 0))
                highlight_point_from_node(map_image, graph, "current", (0, 255, 0))

                cv2.namedWindow("localization", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("localization", 1152, 1152)
                cv2.imshow("localization", map_image)

                if args.save_img:
                    cv2.imwrite(visualization_result_path + os.sep + f"{i:06d}.jpg", map_image)

                cv2.waitKey()

                graph.remove_node("current")

        sim.close()
