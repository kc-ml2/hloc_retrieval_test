import argparse
import random

import cv2
import numpy as np

from algorithms.rrt import RRT
from utils.habitat_utils import display_map, init_map_display

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list-file")
    args, _ = parser.parse_known_args()
    scene_list_file = args.scene_list_file

    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = False

    meters_per_pixel = 0.1
    obstacle_radius = 2

    N_iter = 500
    StepSize = 3

    init_map_display()

    recolored_topdown_map = cv2.imread("./data/recolored_topdown/SN83YJsR3w2_1.bmp", cv2.IMREAD_GRAYSCALE)
    topdown_map = cv2.imread("./data/topdown/SN83YJsR3w2_1.bmp", cv2.IMREAD_GRAYSCALE)

    obstacles_pnt = np.where(topdown_map == 2)
    obstacles_list = []
    for i, y in enumerate(obstacles_pnt[0]):
        pnt = (y, obstacles_pnt[1][i])
        obstacles_list.append(pnt)
    print("Obstacle list generated")

    area_pnt = np.where(topdown_map == 1)
    area_list = []
    for i, y in enumerate(area_pnt[0]):
        pnt = (y, area_pnt[1][i])
        area_list.append(pnt)
    print("Area list generated")

    for i in range(1000):
        root_pnt = random.choice(area_list)
        circle_mask = np.zeros(np.shape(topdown_map), np.uint8)
        circle_mask = cv2.circle(circle_mask, (root_pnt[1], root_pnt[0]), obstacle_radius, 255, -1)
        check_patch = circle_mask * topdown_map
        if 2 in check_patch:
            continue
        break
    if i == 999:
        print("Cannot find a root point for 1000 iteration")
    else:
        print(f"Root point is sampled in {i} steps")

    # root_pnt = (96, 128)

    RRTGraph = RRT(
        startpos=root_pnt,
        obstacles=obstacles_list,
        n_iter=N_iter,
        radius=obstacle_radius,
        stepSize=StepSize,
        area_list=area_list,
    )

    lines = [(RRTGraph.vertices[edge[0]], RRTGraph.vertices[edge[1]]) for edge in RRTGraph.edges]

    converted_lines = []
    for line in lines:
        converted_line = ((line[0][1], line[0][0]), (line[1][1], line[1][0]))
        converted_lines.append(converted_line)

    for line in converted_lines:
        recolored_topdown_map = cv2.line(recolored_topdown_map, line[0], line[1], (0, 255, 0), 1)

    display_map(recolored_topdown_map, wait_for_key=True)
