import cv2
import numpy as np
from skimage.morphology import skeletonize
import sknw


def convert_to_visual_binarymap(topdown_map):
    """Convert habitat map data to 255 binary map.
    Converted map has 255 or 0 value for visualization
    """
    if len(np.shape(topdown_map)) == 3 or np.max(topdown_map) > 2:
        raise ValueError("Input must be the map from habitat.utils.visualization.maps directly")

    _, binary_map = cv2.threshold(topdown_map, 0, 255, cv2.THRESH_BINARY)

    return binary_map


def convert_to_binarymap(topdown_map):
    """Convert habitat map data to 255 binary map.
    Converted map has 255 or 0 value for visualization
    """
    if len(np.shape(topdown_map)) == 3 or np.max(topdown_map) > 2:
        raise ValueError("Input must be the map from habitat.utils.visualization.maps directly")

    _, binary_map = cv2.threshold(topdown_map, 0, 1, cv2.THRESH_BINARY)

    return binary_map


def convert_to_topology(binary_map):
    """Convert binary image to topology."""
    skeleton = skeletonize(binary_map).astype(np.uint8)
    skeleton[skeleton > 0] = 255

    graph = sknw.build_sknw(skeleton)

    return skeleton, graph


def display_graph(map_image, graph, window_name="graph", line_edge=False, wait_for_key=False):
    """Draw nodes and edges into map image."""
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    if line_edge:
        for (s, e) in graph.edges():
            cv2.line(
                img=map_image,
                pt1=(int(graph.nodes[s]["o"][1]), int(graph.nodes[s]["o"][0])),
                pt2=(int(graph.nodes[e]["o"][1]), int(graph.nodes[e]["o"][0])),
                color=(255, 0, 0),
                thickness=1,
            )
    else:
        for (s, e) in graph.edges():
            for pnt in graph[s][e]["pts"]:
                map_image[int(pnt[0])][int(pnt[1])] = (255, 0, 0)
            if graph[s][e]["pts"] == []:
                cv2.line(
                    img=map_image,
                    pt1=(int(graph.nodes[s]["o"][1]), int(graph.nodes[s]["o"][0])),
                    pt2=(int(graph.nodes[e]["o"][1]), int(graph.nodes[e]["o"][0])),
                    color=(255, 0, 0),
                    thickness=1,
                )

    node_points = np.array([graph.nodes()[i]["o"] for i in graph.nodes()])

    for pnt in node_points:
        cv2.circle(
            img=map_image,
            center=(int(pnt[1]), int(pnt[0])),
            radius=1,
            color=(0, 0, 255),
            thickness=-1,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1152, 1152)
    cv2.imshow(window_name, map_image)

    if wait_for_key:
        cv2.waitKey()


def generate_map_image(map_image, graph, line_edge=False):
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    if line_edge:
        for (s, e) in graph.edges():
            cv2.line(
                img=map_image,
                pt1=(int(graph.nodes[s]["o"][1]), int(graph.nodes[s]["o"][0])),
                pt2=(int(graph.nodes[e]["o"][1]), int(graph.nodes[e]["o"][0])),
                color=(255, 0, 0),
                thickness=1,
            )
    else:
        for (s, e) in graph.edges():
            for pnt in graph[s][e]["pts"]:
                map_image[int(pnt[0])][int(pnt[1])] = (255, 0, 0)
            if graph[s][e]["pts"] == []:
                cv2.line(
                    img=map_image,
                    pt1=(int(graph.nodes[s]["o"][1]), int(graph.nodes[s]["o"][0])),
                    pt2=(int(graph.nodes[e]["o"][1]), int(graph.nodes[e]["o"][0])),
                    color=(255, 0, 0),
                    thickness=1,
                )

    node_points = np.array([graph.nodes()[i]["o"] for i in graph.nodes()])

    for pnt in node_points:
        cv2.circle(
            img=map_image,
            center=(int(pnt[1]), int(pnt[0])),
            radius=1,
            color=(0, 0, 255),
            thickness=-1,
        )

    return map_image


def remove_isolated_area(topdown_map, removal_threshold=1000):
    contours, _ = cv2.findContours(topdown_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < removal_threshold:
            cv2.fillPoly(topdown_map, [contour], 0)

    return topdown_map


def prune_graph(graph, topdown_map, check_radius):
    end_node_list = []
    isolated_node_list = []
    root_node_list = []

    for node in graph.nodes:
        if len(list(graph.neighbors(node))) == 1:
            end_node_list.append(node)
        if len(list(graph.neighbors(node))) == 0:
            isolated_node_list.append(node)

    for isolated_node in isolated_node_list:
        graph.remove_node(isolated_node)

    for end_node in end_node_list:
        pnt = [int(graph.nodes[end_node]["o"][0]), int(graph.nodes[end_node]["o"][1])]
        check_patch = topdown_map[
            pnt[0] - check_radius : pnt[0] + check_radius, pnt[1] - check_radius : pnt[1] + check_radius
        ]
        if (2 in check_patch) or (0 in check_patch):
            graph.remove_node(end_node)

    for node in graph.nodes:
        if len(list(graph.neighbors(node))) == 2:
            root_node_list.append(node)

    for root_node in root_node_list:
        branch = list(graph.neighbors(root_node))
        if len(branch) != 2:
            continue
        graph.remove_node(root_node)
        graph.add_edge(branch[0], branch[1])
        graph.edges[branch[0], branch[1]]["pts"] = []

    return graph
