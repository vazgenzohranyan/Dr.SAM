import cv2
import numpy as np
import skimage.morphology
from plantcv import plantcv as pcv
from .utils import remove_pixels_from_skeleton, dialate_erode


def skeletonize(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, list, int]:
    """
    Skeletonize the image.
    :param img: Gray scale image.
    :return: Binary image of the skeleton.
    """
    dialated = dialate_erode(img.copy())

    last_image = dialated.copy()
    r = 0
    for i in range(100):
        if (skimage.morphology.thin(last_image, max_num_iter=1).astype(np.uint8) == last_image).all():
            r = i
            break
        else:
            last_image = skimage.morphology.thin(last_image, max_num_iter=1).astype(np.uint8).copy()

    thinned = remove_pixels_from_skeleton(img, last_image)

    pruned_skeleton, seg_im, segobj = pcv.morphology.prune(skel_img=thinned, size=60)

    manual_pruned = pruned_skeleton.copy()

    for seg in segobj:
        if len(seg) < 60:
            for p in seg:
                p = p[0]
                manual_pruned[p[1]][p[0]] = 1

    return *pcv.morphology.prune(skel_img=manual_pruned, size=60), r - 1


def get_thickness_from_skeleton(img: np.ndarray) -> np.ndarray:
    """
    Get the thickness of the skeleton based on the morphological thinning mechanism.
    :param img: Gray scale image.
    :return: Numpy array representing the thickness of the skeleton.
    """
    distances = np.zeros_like(img)
    skeleton, _, segments, r = skeletonize(img)

    for i in range(0, r + 1):
        thinned = skimage.morphology.thin(dialate_erode(img), max_num_iter=i).astype(np.uint8)

        xored = np.logical_xor(thinned, skeleton).astype(np.uint8)

        closed = dialate_erode(xored)
        skeleton_part = np.logical_xor(closed, thinned).astype(np.uint8)

        distances[skeleton_part == 1] += 1
    distances[distances != 0] = r - distances[distances != 0]

    return distances


def get_thickness_from_distance_transform(img: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Get the thickness of the skeleton based on the distance transform mechanism.
    :param img: Gray scale image.
    :param skeleton: Numpy nd.array representing the skeleton.
    :return: Numpy array representing the thickness of the skeleton.
    """
    distance = cv2.distanceTransform(dialate_erode(img.copy()), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    thickness = cv2.multiply(distance, skeleton.astype(np.float32))
    return thickness
