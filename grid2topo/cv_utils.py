import cv2
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    # Get grid index with interval of 16 pixels
    idx_y, idx_x = np.mgrid[step / 2 : h : step, step / 2 : w : step].astype(np.int64)
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    for x, y in indices:
        # Draw points at each grid index position
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        # Flow value at each grid index
        dx, dy = flow[y, x].astype(np.int64)
        # Draw flow lines at each grid index
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)
