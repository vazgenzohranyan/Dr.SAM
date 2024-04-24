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
