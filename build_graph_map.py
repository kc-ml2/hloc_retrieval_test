import argparse
import itertools
import json
import os

import cv2
import networkx as nx
import numpy as np

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, Cam360Config, PathConfig
from habitat_env.environment import HabitatSimWithMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--obs-path", default="./output/observations/")
    parser.add_argument("--pos-record", default="./output/pos_record.json")
    parser.add_argument("--result-cache", default="./output/similarity_matrix.npy")
    args, _ = parser.parse_known_args()
    height_json_path = args.map_height_json
    obs_path = args.obs_path
    pos_record = args.pos_record
    result_cache = args.result_cache

    # Open files
    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    with open(pos_record, "r") as f:  # pylint: disable=unspecified-encoding
        pos_record = json.load(f)

    with open(result_cache, "rb") as f:  # pylint: disable=unspecified-encoding
        similarity_matrix = np.load(f)

    # Similation initialize
    scene_number = pos_record["scene_number"]
    sim = HabitatSimWithMap(scene_number, Cam360Config, ActionConfig, PathConfig, height_data)

    position = pos_record["000000_sim"]
    sim.update_closest_map(position)

    # Make lists to iteration for building graph
    sorted_obs_image_file = sorted(os.listdir(obs_path))
    obs_id_list = [obs_image_file[:-4] for obs_image_file in sorted_obs_image_file]
    img_extension = sorted_obs_image_file[0][-4:]
    consecutive_edge_list = [(obs_id_list[i], obs_id_list[i + 1]) for i in range(len(obs_id_list) - 1)]

    # In case of visual shortcuts considering only max confidence(similarity probability)
    if TestConstant.VISUAL_SHORTCUT_WITH_MAX_VALUE:
        argmax_similarity_matrix = np.zeros((len(similarity_matrix), len(similarity_matrix)))
        max_row_index = np.argmax(similarity_matrix, axis=0)
        max_column_index = np.argmax(similarity_matrix, axis=1)

        for i in range(len(similarity_matrix)):
            argmax_similarity_matrix[max_row_index[i]][i] = 1
            argmax_similarity_matrix[i][max_column_index[i]] = 1

        similarity_matrix = argmax_similarity_matrix

    # Make similarity dictionary to build graph
    similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
    similarity_list = []
    for combination in similarity_combination_list:
        probability = similarity_matrix[int(combination[0]), int(combination[1])]
        if probability >= 0.99:
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

    # Add edge between consecutive nodes
    for edge in consecutive_edge_list:
        G.edges[edge]["consecutive"] = 1
        G.edges[edge]["similarity"] = 0

    # If visual shortcuts exists, remove consecutive edge & add similarity value
    for similarity in similarity_list:
        G.edges[similarity["edge"]]["consecutive"] = 0
        G.edges[similarity["edge"]]["similarity"] = similarity["probability"]

    # Visualization
    img_id = 0
    for obs_id in obs_id_list:
        map_image = cv2.cvtColor(sim.recolored_topdown_map, cv2.COLOR_GRAY2BGR)
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
