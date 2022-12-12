import argparse
import itertools
import os

import numpy as np
import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from algorithms.sptm_utils import preprocess_image_wo_label
from config.algorithm_config import NetworkConstant, TestConstant
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
            print("Height data is not found.")

        for level in range(num_level):
            print("scene: ", scene_number, "    level: ", level)
            list_to_iterate_by_level = []

            # Set file path
            map_obs_dir = os.path.join(observation_path, f"map_node_observation_level_{level}")
            sample_dir = os.path.join(observation_path, f"test_sample_{level}")

            # Set output npy file name
            cache_index = os.path.basename(os.path.normpath(sample_dir))
            output = os.path.join(observation_path, f"similarity_matrix_{cache_index}.npy")

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_dir))
            sorted_test_sample_file = sorted(os.listdir(sample_dir))
            map_obs_id_list = [map_image_file[:-4] for map_image_file in sorted_map_obs_file]
            test_obs_id_list = [test_image_file[:-4] for test_image_file in sorted_test_sample_file]

            img_extension = sorted_map_obs_file[0][-4:]

            similarity_combination_list = list(itertools.product(map_obs_id_list, test_obs_id_list))

            with tf.device("/device:GPU:0"):
                record_dataset = tf.data.Dataset.from_tensor_slices(similarity_combination_list)
                record_dataset = record_dataset.map(
                    lambda x, m_dir=map_obs_dir, s_dir=sample_dir, ext=img_extension: preprocess_image_wo_label(
                        x, m_dir, s_dir, ext
                    )
                )
                record_dataset = record_dataset.batch(TestConstant.BATCH_SIZE)

                siamese = ResnetBuilder.build_siamese_resnet_18
                model = siamese(
                    (NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS)
                )
                model.load_weights(loaded_model, by_name=True)

                predictions = model.predict(record_dataset)

            # Save similarity matrix that contains similarity probabilities
            similarity_matrix = np.zeros((len(map_obs_id_list), len(test_obs_id_list)))
            for i, combination in enumerate(similarity_combination_list):
                similarity_matrix[int(combination[0])][int(combination[1])] = predictions[i][1]

            with open(output, "wb") as f:
                np.save(f, similarity_matrix)
