import argparse
import itertools
import os

import cv2
import networkx as nx
import numpy as np
import tensorflow as tf

from algorithms.resnet import ResnetBuilder
from algorithms.sptm_utils import preprocess_image_for_building_map
from config.algorithm_config import NetworkConstant, TestConstant, TrainingConstant

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-path", default="./output/observations/")
    parser.add_argument("--load-model", default="./model_weights/model0929_32batch_full_data_93.weights.best.hdf5")
    parser.add_argument("--result-cache", default="./output/similarity_matrix.npy")
    args, _ = parser.parse_known_args()
    obs_path = args.obs_path
    loaded_model = args.load_model
    result_cache = args.result_cache

    sorted_obs_image_file = sorted(os.listdir(obs_path))
    obs_id_list = [obs_image_file[:-4] for obs_image_file in sorted_obs_image_file]
    img_extension = sorted_obs_image_file[0][-4:]
    consecutive_edge_list = [(obs_id_list[i], obs_id_list[i + 1]) for i in range(len(obs_id_list) - 1)]

    similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
    temp_combination_list = similarity_combination_list.copy()

    for combination in temp_combination_list:
        gap_between_nodes = int(combination[1]) - int(combination[0])
        if gap_between_nodes < TrainingConstant.POSITIVE_SAMPLE_DISTANCE:
            similarity_combination_list.remove(combination)

    with tf.device("/device:GPU:0"):
        record_dataset = tf.data.Dataset.from_tensor_slices(similarity_combination_list)
        record_dataset = record_dataset.map(lambda x: preprocess_image_for_building_map(x, obs_path, img_extension))
        record_dataset = record_dataset.batch(TestConstant.BATCH_SIZE)

        siamese = ResnetBuilder.build_siamese_resnet_18
        model = siamese((NetworkConstant.NET_HEIGHT, NetworkConstant.NET_WIDTH, 2 * NetworkConstant.NET_CHANNELS))
        model.load_weights(loaded_model, by_name=True)

        predictions = model.predict(record_dataset)
        predictions = tf.math.argmax(predictions, -1)
        pred_np = predictions.numpy()

    similarity_matrix = np.zeros((len(obs_id_list), len(obs_id_list)))
    for i, combination in enumerate(similarity_combination_list):
        similarity_matrix[int(combination[0])][int(combination[1])] = int(pred_np[i])
        similarity_matrix[int(combination[1])][int(combination[0])] = int(pred_np[i])

    with open(result_cache, "wb") as f:
        np.save(f, similarity_matrix)

    G = nx.Graph()
    G.add_nodes_from(obs_id_list)
    G.add_edges_from(consecutive_edge_list)

    for obs_id in obs_id_list:
        G.nodes[obs_id]["obs"] = cv2.imread(obs_path + os.sep + obs_id + img_extension)

    for edge in consecutive_edge_list:
        G.edges[edge]["type"] = "consecutive"
