import cv2
import numpy as np
import skimage.morphology
from plantcv import plantcv as pcv
from .utils import get_skeleton, remove_pixels_from_skeleton, dialate_erode


def skeletonize_v1(img):
    img = img.copy()
    erodated_skeleton = get_skeleton(dialate_erode(img))

    processed_skeleton = remove_pixels_from_skeleton(img, erodated_skeleton)

    pruned_skeleton, seg_im, segobj = pcv.morphology.prune(skel_img=processed_skeleton, size=80)

    manual_pruned = pruned_skeleton.copy()

    for seg in segobj:
        if len(seg) < 80:
            for p in seg:
                p = p[0]
                manual_pruned[p[1]][p[0]] = 1

    return pcv.morphology.prune(skel_img=manual_pruned, size=80)


def skeletonize_v2(img):
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


def get_thickness_from_skeleton_v2(img):
    distances = np.zeros_like(img)
    skeleton, _, segments, r = skeletonize_v2(img)

    for i in range(0, r + 1):
        thinned = skimage.morphology.thin(dialate_erode(img), max_num_iter=i).astype(np.uint8)

        xored = np.logical_xor(thinned, skeleton).astype(np.uint8)

        closed = dialate_erode(xored)
        skeleton_part = np.logical_xor(closed, thinned).astype(np.uint8)

        distances[skeleton_part == 1] += 1
    distances[distances != 0] = r - distances[distances != 0]

    return distances


def get_thickness_from_skeleton(img, skeleton):
    distance = cv2.distanceTransform(dialate_erode(img.copy()), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    thickness = cv2.multiply(distance, skeleton.astype(np.float32))
    return thickness