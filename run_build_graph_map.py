import argparse
import itertools
import json
import os

import cv2
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import pdist

from config.algorithm_config import TestConstant
from config.env_config import ActionConfig, CamFourViewConfig, PathConfig
from habitat_env.environment import HabitatSimWithMap
from utils.habitat_utils import draw_line_from_edge, draw_point_from_node, highlight_point_from_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-height-json", default="./data/map_height.json")
    parser.add_argument("--obs-path", required=True, help="directory containing observation images")
    parser.add_argument("--pos-record")
    parser.add_argument("--result-cache")
    parser.add_argument("--save-img", action="store_true")
    args, _ = parser.parse_known_args()
    height_json_path = args.map_height_json
    obs_path = args.obs_path

    # Open position record file to display result on map (From argument or from pre-defined name)
    cache_index = os.path.basename(os.path.normpath(obs_path))
    parent_dir = os.path.dirname(os.path.dirname(obs_path))
    if args.pos_record:
        pos_record = args.pos_record
    else:
        pos_record = os.path.join(parent_dir, f"pos_record_{cache_index}.json")

    # Open similarity result cache files to build graph (From argument or from pre-defined name)
    if args.result_cache:
        result_cache = args.result_cache
    else:
        result_cache = os.path.join(parent_dir, f"similarity_matrix_{cache_index}.npy")

    if args.save_img:
        visualization_result_path = os.path.join(parent_dir, f"similarity_visualize_result_{cache_index}")
        os.makedirs(visualization_result_path, exist_ok=True)

    # Open files
    with open(height_json_path, "r") as height_json:  # pylint: disable=unspecified-encoding
        height_data = json.load(height_json)

    with open(pos_record, "r") as f:  # pylint: disable=unspecified-encoding
        pos_record = json.load(f)

    with open(result_cache, "rb") as f:  # pylint: disable=unspecified-encoding
        similarity_matrix = np.load(f)

    # Similation initialize
    scene_number = pos_record["scene_number"]
    sim = HabitatSimWithMap(scene_number, CamFourViewConfig, ActionConfig, PathConfig, height_data)

    position = pos_record["000000_sim"]
    sim.update_closest_map(position)

    # Make lists to iteration for building graph
    sorted_obs_image_file = sorted(os.listdir(obs_path))
    obs_id_list = [obs_image_file[:-4] for obs_image_file in sorted_obs_image_file]
    img_extension = sorted_obs_image_file[0][-4:]
    consecutive_edge_id_list = [(obs_id_list[i], obs_id_list[i + 1]) for i in range(len(obs_id_list) - 1)]

    # Make visual shortcut dictionary list to build graph
    visual_shortcut_list = []
    similarity_combination_list = list(itertools.combinations(obs_id_list, 2))
    for combination in similarity_combination_list:
        similarity = similarity_matrix[int(combination[0]), int(combination[1])]
        if similarity >= TestConstant.SIMILARITY_PROBABILITY_THRESHOLD:
            visual_shortcut_list.append({"edge_id": combination, "similarity": similarity})

    # Make linkage for clustering
    linkage_dataset = similarity_matrix >= TestConstant.SIMILARITY_PROBABILITY_THRESHOLD
    linkage_list = []
    for index, data in np.ndenumerate(linkage_dataset):
        if data:
            linkage_list.append([index[0], index[1]])
    distance_matrix = pdist(linkage_list)
    dendrogram = ward(distance_matrix)

    # Assign cluster number to each obsercation id
    cluster_assign_list = fcluster(dendrogram, t=500, criterion="distance")
    # cluster_assign_list = fcluster(dendrogram, t=0.5)
    print("The number of clusters : ", np.max(cluster_assign_list))
    cluster_table = np.zeros([len(obs_id_list), np.max(cluster_assign_list)], dtype=np.int32)
    for i, linkage in enumerate(linkage_list):
        s, e = linkage
        cluster_table[s][cluster_assign_list[i] - 1] = 1
        cluster_table[e][cluster_assign_list[i] - 1] = 1

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from(obs_id_list)
    G.add_edges_from(consecutive_edge_id_list)
    G.add_edges_from([shortcut["edge_id"] for shortcut in visual_shortcut_list])

    # Put values into graph nodes & edges
    # for obs_id in obs_id_list:
    #     G.nodes[obs_id]["obs"] = cv2.imread(obs_path + os.sep + obs_id + img_extension)

    for i in G.nodes():
        G.nodes()[i]["o"] = pos_record[i + "_grid"]

    # Add edge between consecutive nodes
    for edge_id in consecutive_edge_id_list:
        G.edges[edge_id]["consecutive"] = 1
        G.edges[edge_id]["similarity"] = 0

    # If visual shortcuts exists, remove consecutive edge & add similarity value
    for shortcut in visual_shortcut_list:
        G.edges[shortcut["edge_id"]]["consecutive"] = 0
        G.edges[shortcut["edge_id"]]["similarity"] = shortcut["similarity"]

    # Visualization
    img_id = 0
    for obs_id in obs_id_list:
        map_image = cv2.cvtColor(sim.recolored_topdown_map, cv2.COLOR_GRAY2BGR)

        for i in G.nodes():
            draw_point_from_node(map_image, G, i)

        # Get observation id numbers which is in the same cluster
        cluster_list = np.where(cluster_table[int(obs_id)] == 1)[0]
        obs_id_in_same_cluster = []
        for cluster in cluster_list:
            obs_in_cluster = np.where(cluster_table[:, cluster] == 1)[0]
            obs_id_in_same_cluster = np.concatenate([obs_id_in_same_cluster, obs_in_cluster])
            obs_id_in_same_cluster = np.int32(obs_id_in_same_cluster)
        obs_id_in_same_cluster = [*set(obs_id_in_same_cluster)]
        obs_id_in_same_cluster = [f"{id:06d}" for id in obs_id_in_same_cluster]
        print("All forward : ", f"{len(obs_id_in_same_cluster) + np.max(cluster_assign_list):06d}")

        # Mark nodes which is in the same cluster
        for id_same_cluster in obs_id_in_same_cluster:
            highlight_point_from_node(map_image, G, id_same_cluster, (0, 122, 255))

        # Draw all shortcuts
        shortcuts_to_draw = []
        for shortcut in visual_shortcut_list:
            if obs_id in shortcut["edge_id"]:
                shortcuts_to_draw.append(shortcut)

        for shortcut in shortcuts_to_draw:
            draw_line_from_edge(map_image, G, shortcut["edge_id"], shortcut["similarity"])

        # Mark the shortcut node with maximum value
        id_max_value = np.argmax([shortcut["similarity"] for shortcut in shortcuts_to_draw])
        (s, e) = shortcuts_to_draw[id_max_value]["edge_id"]
        id_max_shortcut = e if s == obs_id else s
        highlight_point_from_node(map_image, G, id_max_shortcut, (0, 255, 255))
        # Mark the current observation node
        highlight_point_from_node(map_image, G, obs_id, (0, 255, 0))

        cv2.namedWindow("visual_shortcut", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("visual_shortcut", 1152, 1152)
        cv2.imshow("visual_shortcut", map_image)

        if args.save_img:
            cv2.imwrite(visualization_result_path + os.sep + f"{img_id:06d}.jpg", map_image)
            img_id = img_id + 1

        cv2.waitKey()
