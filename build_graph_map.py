import argparse
import itertools
import json
import os

import cv2
import networkx as nx
import numpy as np

from config.env_config import ActionConfig, Cam360Config, PathConfig
from utils.habitat_utils import get_closest_map, get_map_from_database, initialize_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file", default="./data/scene_list_test.txt")
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--obs-path", default="./output/observations/")
    parser.add_argument("--pos-record", default="./output/pos_record.json")
    parser.add_argument("--result-cache", default="./output/similarity_matrix.npy")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file
    height_json_path = args.map_height_json
    obs_path = args.obs_path
    pos_record = args.pos_record
    result_cache = args.result_cache

    with open(scene_list_file) as f:  # pylint: disable=unspecified-encoding
        scene_list = f.read().splitlines()

    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    with open(pos_record, "r") as f:  # pylint: disable=unspecified-encoding
        pos_record = json.load(f)

    with open(result_cache, "rb") as f:  # pylint: disable=unspecified-encoding
        similarity_matrix = np.load(f)

    scene_number = scene_list[0]
    sim = initialize_sim(scene_number, Cam360Config, ActionConfig, PathConfig)
    recolored_topdown_map_list, _, _ = get_map_from_database(scene_number, height_data)

    position = pos_record["000000_sim"]
    recolored_topdown_map, closest_level = get_closest_map(sim, position, recolored_topdown_map_list)

    sorted_obs_image_file = sorted(os.listdir(obs_path))
    obs_id_list = [obs_image_file[:-4] for obs_image_file in sorted_obs_image_file]
    img_extension = sorted_obs_image_file[0][-4:]
    consecutive_edge_list = [(obs_id_list[i], obs_id_list[i + 1]) for i in range(len(obs_id_list) - 1)]

    similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
    similarity_edge_list = []
    for combination in similarity_combination_list:
        if similarity_matrix[int(combination[0]), int(combination[1])] == 1:
            similarity_edge_list.append(combination)

    G = nx.Graph()
    G.add_nodes_from(obs_id_list)
    G.add_edges_from(consecutive_edge_list)
    G.add_edges_from(similarity_edge_list)

    # for obs_id in obs_id_list:
    #     G.nodes[obs_id]["obs"] = cv2.imread(obs_path + os.sep + obs_id + img_extension)

    for i in G.nodes():
        G.nodes()[i]["o"] = pos_record[i + "_grid"]

    for edge in consecutive_edge_list:
        G.edges[edge]["consecutive"] = 1
        G.edges[edge]["similarity"] = 0

    for edge in similarity_edge_list:
        G.edges[edge]["consecutive"] = 0
        G.edges[edge]["similarity"] = 1

    for edge in similarity_edge_list:
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

        (s, e) = edge
        if G.edges[(s, e)]["similarity"] == 1:
            cv2.line(
                img=map_image,
                pt1=(int(G.nodes[s]["o"][1]), int(G.nodes[s]["o"][0])),
                pt2=(int(G.nodes[e]["o"][1]), int(G.nodes[e]["o"][0])),
                color=(255, 0, 0),
                thickness=1,
            )

        cv2.namedWindow("similarity", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("similarity", 1152, 1152)
        cv2.imshow("similarity", map_image)
        cv2.waitKey()
