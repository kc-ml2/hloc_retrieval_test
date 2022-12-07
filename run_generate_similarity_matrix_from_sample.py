import argparse
import itertools
import os

import numpy as np
import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from algorithms.sptm_utils import preprocess_image_for_localization
from config.algorithm_config import NetworkConstant, TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", default="./model_weights/model.20221129-125905.32batch.4view.weights.best.hdf5")
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--map-obs-path", default="./output")
    parser.add_argument("--output")
    args, _ = parser.parse_known_args()
    loaded_model = args.load_model
    scene_list_file = args.scene_list_file
    scene_index = args.scene_index
    map_obs_path = args.map_obs_path

    # Open files
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    if scene_index is not None:
        scene_list = [scene_list[scene_index]]

    for scene_number in scene_list:
        sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig)
        observation_path = os.path.join(map_obs_path, f"observation_{scene_number}")

        for level, recolored_topdown_map in enumerate(sim.recolored_topdown_map_list):
            print("scene: ", scene_number, "    level: ", level)

            # Set file path
            map_obs_path = os.path.join(observation_path, f"map_node_observation_level_{level}")
            test_sample_path = os.path.join(observation_path, f"test_sample_{level}")
            path_pair = [map_obs_path, test_sample_path]

            # Set output npy file name
            if args.output:
                output = args.output
            else:
                cache_index = os.path.basename(os.path.normpath(test_sample_path))
                output = os.path.join(observation_path, f"similarity_matrix_{cache_index}.npy")

            # Make list to iterate
            sorted_map_obs_file = sorted(os.listdir(map_obs_path))
            sorted_test_sample_file = sorted(os.listdir(test_sample_path))
            map_obs_id_list = [map_image_file[:-4] for map_image_file in sorted_map_obs_file]
            test_obs_id_list = [test_image_file[:-4] for test_image_file in sorted_test_sample_file]
            map_img_extension = sorted_map_obs_file[0][-4:]
            test_img_extension = sorted_test_sample_file[0][-4:]
            extension_pair = [map_img_extension, test_img_extension]

            similarity_combination_list = list(itertools.product(map_obs_id_list, test_obs_id_list))

            with tf.device("/device:GPU:1"):
                record_dataset = tf.data.Dataset.from_tensor_slices(similarity_combination_list)
                record_dataset = record_dataset.map(
                    lambda x: preprocess_image_for_localization(x, path_pair, extension_pair)
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

        sim.close()
