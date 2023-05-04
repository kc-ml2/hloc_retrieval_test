import argparse
import os

import numpy as np
import tensorflow as tf

from network.resnet import ResnetBuilder
from relocalization.localization_realworld import LocalizationRealWorld
from utils.config_import import load_config_module
from utils.network_utils import preprocess_single_image_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/realworld_69FOV.py")
    args, _ = parser.parse_known_args()
    module_name = args.config

    config = load_config_module(module_name)
    map_obs_path = config.PathConfig.LOCALIZATION_TEST_PATH
    loaded_model = config.PathConfig.MODEL_WEIGHTS

    total_list_to_iterate = []
    output_size_list = []

    with tf.device(f"/device:GPU:{config.PathConfig.GPU_ID}"):
        model, top_network, bottom_network = ResnetBuilder.load_siamese_model(loaded_model, config.NetworkConstant)

    # Set file path
    map_cache_index = "map_node_observation_level_0"
    query_cache_index = "test_query_0"
    map_obs_dir = os.path.join(map_obs_path, map_cache_index)
    query_dir = os.path.join(map_obs_path, query_cache_index)

    # Make list to iterate
    sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
    sorted_test_query_file = sorted(os.listdir(query_dir))
    map_obs_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_file]
    query_list = [query_dir + os.sep + file for file in sorted_test_query_file]

    # Set output npy file name
    localization = LocalizationRealWorld(
        config, top_network, bottom_network, map_obs_dir, query_dir=query_dir, instance_only=True
    )
    map_output = localization.map_embedding_file
    query_output = localization.query_embedding_file

    # Append to total iteration list
    total_list_to_iterate = total_list_to_iterate + map_obs_list
    output_size_list.append((map_output, len(sorted_map_obs_file)))
    total_list_to_iterate = total_list_to_iterate + query_list
    output_size_list.append((query_output, len(sorted_test_query_file)))

    with tf.device(f"/device:GPU:{config.PathConfig.GPU_ID}"):
        record_dataset = tf.data.Dataset.from_tensor_slices(total_list_to_iterate)
        record_dataset = record_dataset.map(lambda x: preprocess_single_image_file(x))
        record_dataset = record_dataset.batch(config.TestConstant.BATCH_SIZE)

        predictions = bottom_network.predict(record_dataset)

    # Save siamese embedding
    index = 0
    for output in output_size_list:
        output_name, output_size = output
        end_index = index + output_size
        embedding = predictions[index:end_index]
        print("File name: ", output_name)
        print("File size: ", np.shape(embedding))
        print("File index: ", f"[{index}:{end_index}]")

        index = end_index

        with open(output_name, "wb") as f:
            np.save(f, embedding)