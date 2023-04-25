import argparse
import os

import numpy as np
import tensorflow as tf

from config.algorithm_config import TestConstant
from config.env_config import PathConfig
from network.resnet import ResnetBuilder
from relocalization.localization_realworld import LocalizationRealWorld
from utils.network_utils import preprocess_single_image_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", default="./model_weights/model.20230410-153553.threeview.69FOV.weights.hdf5")
    parser.add_argument("--map-obs-path", default="./output_realworld")
    args, _ = parser.parse_known_args()
    loaded_model = args.load_model
    map_obs_path = args.map_obs_path

    total_list_to_iterate = []
    output_size_list = []

    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        model, top_network, bottom_network = ResnetBuilder.load_siamese_model(loaded_model)

    # Set file path
    map_cache_index = "map_node_observation_level_0"
    sample_cache_index = "test_sample_0"
    map_obs_dir = os.path.join(map_obs_path, map_cache_index)
    sample_dir = os.path.join(map_obs_path, sample_cache_index)

    # Make list to iterate
    sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
    sorted_test_sample_file = sorted(os.listdir(sample_dir))
    map_obs_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_file]
    sample_list = [sample_dir + os.sep + file for file in sorted_test_sample_file]

    # Set output npy file name
    localization = LocalizationRealWorld(
        top_network, bottom_network, map_obs_dir, sample_dir=sample_dir, instance_only=True, num_frames_per_node=1
    )
    map_output = localization.map_embedding_file
    sample_output = localization.sample_embedding_file

    # Append to total iteration list
    total_list_to_iterate = total_list_to_iterate + map_obs_list
    output_size_list.append((map_output, len(sorted_map_obs_file)))
    total_list_to_iterate = total_list_to_iterate + sample_list
    output_size_list.append((sample_output, len(sorted_test_sample_file)))

    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        record_dataset = tf.data.Dataset.from_tensor_slices(total_list_to_iterate)
        record_dataset = record_dataset.map(lambda x: preprocess_single_image_file(x))
        record_dataset = record_dataset.batch(TestConstant.BATCH_SIZE)

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
