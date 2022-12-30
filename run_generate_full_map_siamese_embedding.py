import argparse
import os

import numpy as np
import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from algorithms.sptm_utils import preprocess_single_image_file
from config.algorithm_config import NetworkConstant, TestConstant
from config.env_config import PathConfig
from utils.habitat_utils import open_env_related_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", default="./model_weights/model.20221129-125905.32batch.4view.weights.best.hdf5")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--map-obs-path", default="./output")
    args, _ = parser.parse_known_args()
    loaded_model = args.load_model
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    height_json_path = args.map_height_json
    map_obs_path = args.map_obs_path

    total_list_to_iterate = []
    output_size_list = []

    # Open files
    scene_list, height_data = open_env_related_files(scene_list_file, height_json_path, scene_index)

    # Make list to iterate for Siamese forward
    for scene_number in scene_list:
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        # Find number of levels
        num_level = 0
        for height in height_data:
            if scene_number in height:
                num_level = num_level + 1
        if num_level == 0:
            raise ValueError("Height data is not found.")

        for level in range(num_level):
            list_to_iterate_by_level = []
            print("scene: ", scene_number, "    level: ", level)

            # Set file path
            map_cache_index = f"map_node_observation_level_{level}"
            sample_cache_index = f"test_sample_{level}"
            map_obs_dir = os.path.join(observation_path, map_cache_index)
            sample_dir = os.path.join(observation_path, sample_cache_index)

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            sorted_test_sample_file = sorted(os.listdir(sample_dir))
            map_obs_list = [map_obs_dir + os.sep + file for file in sorted_map_obs_file]
            sample_list = [sample_dir + os.sep + file for file in sorted_test_sample_file]

            # Set output npy file name & expected size in prediction list
            map_output = os.path.join(observation_path, f"siamese_embedding_{map_cache_index}.npy")
            sample_output = os.path.join(observation_path, f"siamese_embedding_{sample_cache_index}.npy")

            # Append to total iteration list
            total_list_to_iterate = total_list_to_iterate + map_obs_list
            output_size_list.append((map_output, len(sorted_map_obs_file)))
            total_list_to_iterate = total_list_to_iterate + sample_list
            output_size_list.append((sample_output, len(sorted_test_sample_file)))

    with tf.device(f"/device:GPU:{PathConfig.GPU_ID}"):
        record_dataset = tf.data.Dataset.from_tensor_slices(total_list_to_iterate)
        record_dataset = record_dataset.map(lambda x: preprocess_single_image_file(x))
        record_dataset = record_dataset.batch(TestConstant.BATCH_SIZE)

        siamese = ResnetBuilder.build_siamese_resnet_18
        model = siamese((NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS))
        model.load_weights(loaded_model, by_name=True)

        bottom_network = ResnetBuilder.build_bottom_network(
            model,
            (NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, NetworkConstant.NET_CHANNELS),
        )

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
