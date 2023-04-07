import argparse
import os

import numpy as np
import tensorflow as tf

from config.env_config import PathConfig
from network.resnet import ResnetBuilder
from relocalization.single_view_localization_realworld import SingleViewLocalizationRealWorld

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-obs-path", default="./output_realworld")
    parser.add_argument("--load-model", default="./model_weights/model.20230331-142500.singleview.69FOV.weights.hdf5")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args, _ = parser.parse_known_args()
    map_obs_path = args.map_obs_path
    loaded_model = args.load_model
    is_sparse = args.sparse
    is_visualize = args.visualize

    # Load pre-trained model & top network
    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        model, top_network, bottom_network = ResnetBuilder.load_siamese_model(loaded_model)

    # Main loop
    total_accuracy = []
    total_d1 = []
    total_d2 = []
    total_samples = 0

    # Set file path
    map_obs_dir = os.path.join(map_obs_path, "map_node_observation_level_0")
    sample_dir = os.path.join(map_obs_path, "test_sample_0")

    localization = SingleViewLocalizationRealWorld(
        top_network,
        bottom_network,
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
