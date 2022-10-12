import argparse
import itertools
import json
import os

import cv2
import networkx as nx
import numpy as np

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, Cam360Config, PathConfig
from utils.habitat_utils import get_closest_map, get_map_from_database, initialize_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--obs-path", default="./output/large_observations/")
    parser.add_argument("--pos-record", default="./output/large_pos_record.json")
    parser.add_argument("--result-cache", default="./output/large_prob_similarity_matrix.npy")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json
    obs_path = args.obs_path
    pos_record = args.pos_record
    result_cache = args.result_cache

    # Open files
    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    with open(pos_record, "r") as f:  # pylint: disable=unspecified-encoding
        pos_record = json.load(f)

    with open(result_cache, "rb") as f:  # pylint: disable=unspecified-encoding
        similarity_matrix = np.load(f)

    # Similation initialize
    scene_number = scene_list[2]
    sim = initialize_sim(scene_number, Cam360Config, ActionConfig, PathConfig)
    recolored_topdown_map_list, _, _ = get_map_from_database(scene_number, height_data)
    position = pos_record["000000_sim"]
    recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)

    # Make lists for node & edge generation
    sorted_obs_image_file = sorted(os.listdir(obs_path))
    obs_id_list = [obs_image_file[:-4] for obs_image_file in sorted_obs_image_file]
    img_extension = sorted_obs_image_file[0][-4:]
    consecutive_edge_list = [(obs_id_list[i], obs_id_list[i + 1]) for i in range(len(obs_id_list) - 1)]

    if TestConstant.NORMALIZE_PROBABILITY:
        # Get lists of  possibilites that are higher than 0.5 & their indices
        idx_list_over_threshold = []
        probability_list_over_threshold = []
        for idx, probability in np.ndenumerate(similarity_matrix):
            if probability > 0.5:
                probability_list_over_threshold.append(probability)
                idx_list_over_threshold.append(idx)

        probability_list_over_threshold = probability_list_over_threshold / np.linalg.norm(
            probability_list_over_threshold
        )
        for i, idx in enumerate(idx_list_over_threshold):
            similarity_matrix[idx] = probability_list_over_threshold[i]

        # Generate similarity list for adding edge
        similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
        similarity_list = []
        for combination in similarity_combination_list:
            if (int(combination[0]), int(combination[1])) in idx_list_over_threshold:  # If probability is over 0.5
                probability = similarity_matrix[int(combination[0]), int(combination[1])]
                similarity_list.append({"edge": combination, "probability": probability})

    else:
        similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
        similarity_list = []
        for combination in similarity_combination_list:
            probability = similarity_matrix[int(combination[0]), int(combination[1])]
            if probability >= 0.5:
                similarity_list.append({"edge": combination, "probability": probability})

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from(obs_id_list)
    G.add_edges_from(consecutive_edge_list)
    G.add_edges_from([similarity["edge"] for similarity in similarity_list])

    # Put values into graph nodes & edges
    # for obs_id in obs_id_list:
    #     G.nodes[obs_id]["obs"] = cv2.imread(obs_path + os.sep + obs_id + img_extension)

    for i in G.nodes():
        G.nodes()[i]["o"] = pos_record[i + "_grid"]

    for edge in consecutive_edge_list:
        G.edges[edge]["consecutive"] = 1
        G.edges[edge]["similarity"] = 0

    for similarity in similarity_list:
        G.edges[similarity["edge"]]["consecutive"] = 0
        G.edges[similarity["edge"]]["similarity"] = similarity["probability"]

    # Visualization
    img_id = 0
    for obs_id in obs_id_list:
        map_image = cv2.cvtColor(recolored_topdown_map, cv2.COLOR_GRAY2BGR)
        node_points = np.array([G.nodes()[i]["o"] for i in G.nodes()])

        for pnt in node_points:
            cv2.circle(
                img=map_image,
                center=(int(pnt[1]), int(pnt[0])),
                radius=0,
                color=(0, 0, 255),
                thickness=-1,
            )

        for similarity in similarity_list:
            if obs_id in similarity["edge"]:
                (s, e) = similarity["edge"]
                cv2.line(
                    img=map_image,
                    pt1=(int(G.nodes[s]["o"][1]), int(G.nodes[s]["o"][0])),
                    pt2=(int(G.nodes[e]["o"][1]), int(G.nodes[e]["o"][0])),
                    color=(int(255 * similarity["probability"]), 0, 0),
                    thickness=1,
                )

        cv2.circle(
            img=map_image,
            center=(int(G.nodes[obs_id]["o"][1]), int(G.nodes[obs_id]["o"][0])),
            radius=1,
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.namedWindow("similarity", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("similarity", 1152, 1152)
        cv2.imshow("similarity", map_image)
        cv2.imwrite(f"./output/large_prob_sim_result/{img_id:06d}.jpg", map_image)
        img_id = img_id + 1
        cv2.waitKey()
