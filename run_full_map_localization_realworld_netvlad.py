import argparse
import os

import numpy as np

from relocalization.localization_realworld_netvlad import LocalizationRealWorldNetVLAD
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

    # Main loop
    total_accuracy = []
    total_d1 = []
    total_d2 = []
    total_samples = 0

    # Set file path
    map_obs_dir = os.path.join(map_obs_path, "map_node_observation_level_0")
    sample_dir = os.path.join(map_obs_path, "test_sample_0")

    localization = LocalizationRealWorldNetVLAD(
        config,
        map_obs_dir,
        sample_dir=sample_dir,
        sparse_map=is_sparse,
        visualize=is_visualize,
    )

    accuracy_list, d1_list, d2_list, num_samples = localization.iterate_localization_with_sample()

    total_accuracy = total_accuracy + accuracy_list
    total_d1 = total_d1 + d1_list
    total_d2 = total_d2 + d2_list
    total_samples = total_samples + num_samples

    print("Accuracy: ", sum(total_accuracy) / total_samples)
    print("Accuracy std: ", np.std(total_accuracy))
    print("Distance 1: ", (sum(total_d1) / total_samples) * 0.1)
    print("Distance 1 std: ", np.std(total_d1) * 0.1)
    print("Distance 2: ", (sum(total_d2) / total_samples) * 0.1)
    print("Distance 2 std: ", np.std(total_d2) * 0.1)
