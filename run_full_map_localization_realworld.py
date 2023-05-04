import argparse
import os

import numpy as np
import tensorflow as tf

from network.resnet import ResnetBuilder
from relocalization.localization_realworld import LocalizationRealWorld
from utils.config_import import load_config_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/realworld_69FOV.py")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    is_sparse = args.sparse
    is_visualize = args.visualize

    config = load_config_module(module_name)
    map_obs_path = config.PathConfig.LOCALIZATION_TEST_PATH
    loaded_model = config.PathConfig.MODEL_WEIGHTS

    # Load pre-trained model & top network
    with tf.device(f"/device:GPU:{config.PathConfig.GPU_ID}"):
        model, top_network, bottom_network = ResnetBuilder.load_siamese_model(loaded_model, config.NetworkConstant)

    # Main loop
    total_accuracy = []
    total_d1 = []
    total_d2 = []
    total_queries = 0

    # Set file path
    map_obs_dir = os.path.join(map_obs_path, "map_node_observation_level_0")
    query_dir = os.path.join(map_obs_path, "test_query_0")

    localization = LocalizationRealWorld(
        config,
        top_network,
        bottom_network,
        map_obs_dir,
        query_dir=query_dir,
        sparse_map=is_sparse,
        visualize=is_visualize,
    )

    # localization = OrbMatchingLocalizationRealWorld(
    #     config,
    #     map_obs_dir=map_obs_dir,
    #     query_dir=query_dir,
    #     sparse_map=is_sparse,
    #     visualize=is_visualize,
    # )

    accuracy_list, d1_list, d2_list, num_queries = localization.iterate_localization_with_query()

    total_accuracy = total_accuracy + accuracy_list
    total_d1 = total_d1 + d1_list
    total_d2 = total_d2 + d2_list
    total_queries = total_queries + num_queries

    print("Accuracy: ", sum(total_accuracy) / total_queries)
    print("Accuracy std: ", np.std(total_accuracy))
    print("Distance 1: ", (sum(total_d1) / total_queries) * 0.1)
    print("Distance 1 std: ", np.std(total_d1) * 0.1)
    print("Distance 2: ", (sum(total_d2) / total_queries) * 0.1)
    print("Distance 2 std: ", np.std(total_d2) * 0.1)
