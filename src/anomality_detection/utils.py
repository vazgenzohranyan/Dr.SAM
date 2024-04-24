import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def dialate_erode(img: np.ndarray, ksize: int = 3, i: int = 1) -> np.ndarray:
    """
    Dialate and erode the image.
    :param img: Image representing Numpy array
    :param ksize: Kernel size
    :param i: Iterations
    :return: Dialated and eroded image.
    """
    processed_img = cv2.dilate(img, np.ones((ksize, ksize)), i)
    processed_img = cv2.erode(processed_img, np.ones((ksize, ksize)), i)
    return processed_img


def draw_circles_from_segments(img: np.ndarray,
                               thickness: np.ndarray,
                               segments: list,
                               step: int = 8,
                               fill: bool = False):
    """
    Draw circles from segments of returned from skeletonization.
    :param img: Grayscale image
    :param thickness: Numpy array representing thickness of the skeleton pixels
    :param segments: Skeleton branches
    :param step: Step size to draw circles
    :param fill: Indicates whether to fill the circle or not
    :return: New image with circles drawn on it.
    """
    new_image = img.copy()
    fill = -1 if fill else 1

    for seg in segments:
        for p in seg[::step]:
            p = p[0]
            new_image = cv2.circle(new_image, (p[0], p[1]), round(thickness[p[1]][p[0]]), (0, 0, 255), fill)

    return new_image


def remove_pixels_from_skeleton(img: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Remove pixels from the skeleton which are not in the original image.
    :param img: Grayscale image
    :param skeleton: Numpy array representing the skeleton
    :return: New skeleton image without pixels not in the original image.
    """
    return np.logical_and(img, skeleton).astype(np.uint8)


def add_points_to_image(img: np.ndarray,
                        points: list,
                        color: tuple = (0, 0, 222),
                        add_change: bool = False) -> np.ndarray:
    """
    Add points to the image drawing circles.
    :param img: Grayscale image
    :param points: List of coordinates of the points
    :param color: RGB shape color of the circles
    :param add_change: Whether to add change percentage to the image or not
    :return: Numpy array representing the image with circles drawn on it.
    """
    _im = img.copy()
    for a in points:
        p = a['p']
        change = a['change']
        _im = cv2.circle(_im, (p[0], p[1]), 5, color, -1)

        if add_change:
            _im = cv2.putText(_im, f"{int(change)}%", (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 1, cv2.LINE_AA)
    return _im


def find_closest_point_from_skeleton(points, skeleton_segments):
    anomality_indicator = [0] * len(skeleton_segments)
    skeleton_points = []

    for p in points:
        x, y = p
        POINT = None
        MIN_DIST = 1e10
        SEGMENT = None

        for i, seg in enumerate(skeleton_segments):
            if len(seg) < 6:
                continue

            segment = np.squeeze(np.array(seg[:len(seg) // 2]))

            dist = np.linalg.norm(np.array([y, x] - segment), axis=1)

            if np.min(dist) < MIN_DIST:
                SEGMENT = i
                POINT = segment[np.argmin(dist)]
                MIN_DIST = np.min(dist)
        skeleton_points.append(POINT)
        anomality_indicator[SEGMENT] = True

    return skeleton_points, anomality_indicator


def get_anomality_centers_from_image(img):
    image = img.copy()
    _image = np.zeros_like(image[:, :, 0])
    high_anomalities = []
    low_anomalities = []

    anomality_high_colors = [[62, 255, 2],
                             [61, 255, 0],
                             [63, 255, 5],
                             [63, 254, 3],
                             [63, 253, 4],
                             [62, 255, 1],
                             [62, 254, 5],
                             [62, 255, 4],
                             [61, 255, 2],
                             [61, 255, 5],
                             [61, 254, 3],
                             [63, 255, 1],
                             [64, 255, 1],
                             [64, 255, 4],
                             [61, 255, 1],
                             [63, 255, 3],
                             [61, 255, 4],
                             [61, 253, 6]]
    anomality_low_colors = [[31, 0, 254],
                            [30, 0, 255],
                            [29, 1, 255],
                            [30, 2, 246],
                            [29, 0, 253],
                            [28, 1, 249],
                            [34, 0, 254],
                            [32, 0, 255],
                            [13, 0, 116],
                            [28, 0, 254],
                            [29, 1, 251],
                            [29, 1, 254],
                            [27, 1, 255],
                            [29, 0, 255],
                            [13, 0, 115],
                            [33, 0, 254],
                            [12, 1, 117],
                            [30, 1, 248],
                            [30, 1, 254],
                            [29, 0, 254],
                            [28, 2, 246],
                            [28, 0, 255],
                            [34, 0, 255],
                            [16, 0, 117]]

    mask = []
    for i in image:
        _m = []
        for j in i:
            if j.tolist() in anomality_high_colors:
                _m.append(True)
            else:
                _m.append(False)
        mask.append(_m)
    mask = np.array(mask).astype(np.uint8)

    contours, hierarchies = cv2.findContours(dialate_erode(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            high_anomalities.append((cy, cx))

    mask = []
    for i in image:
        _m = []
        for j in i:
            if j.tolist() in anomality_low_colors:
                _m.append(True)
            else:
                _m.append(False)
        mask.append(_m)
    mask = np.array(mask).astype(np.uint8)

    contours, hierarchies = cv2.findContours(dialate_erode(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            low_anomalities.append((cy, cx))

    return {'high': high_anomalities, 'low': low_anomalities}
