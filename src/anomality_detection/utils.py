import cv2
import math
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def plot_over_image(img1, img2):
    plot_image(np.logical_xor(img1, img2))


def dialate_erode(img, ksize=3, i=1):
    processed_img = cv2.dilate(img, np.ones((ksize, ksize)), i)
    processed_img = cv2.erode(processed_img, np.ones((ksize, ksize)), i)
    return processed_img


def get_perp_coord(aX, aY, bX, bY, length=40):
    vX = bX - aX
    vY = bY - aY
    # print(str(vX)+" "+str(vY))
    if (vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX * vX + vY * vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0 - vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)


def get_skeleton(img):
    return pcv.morphology.skeletonize(img)


def prune(skeleton, size=80):
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton, size=40)
    return pruned_skeleton / 255


def get_distance(img):
    distance = cv2.distanceTransform(img.copy(), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    return distance


def draw_circles(img, thickness):
    new_image = img.copy()
    for i in range(0, thickness.shape[0], 20):
        for j in range(thickness.shape[1]):
            if thickness[i][j] > 0:
                new_image = cv2.circle(new_image, (j, i), round(thickness[i][j]), (0, 0, 255), 1)
    return new_image


def draw_circles_from_segments(img, thickness, segments, step=8, fill=False):
    new_image = img.copy()
    fill = -1 if fill else 1

    for seg in segments:
        for p in seg[::step]:
            p = p[0]
            new_image = cv2.circle(new_image, (p[0], p[1]), round(thickness[p[1]][p[0]]), (0, 0, 255), fill)

    return new_image


def draw_lines_from_segments(img, thickness, segments, step=8):
    new_image = img.copy()

    for seg in segments:
        last_point = seg[0][0]
        for p in seg[1::step]:
            p = p[0]

            x1, x2, y1, y2 = get_perp_coord(p[1], p[0], last_point[1], last_point[0])
            new_image = cv2.line(new_image, (x2, x1), (y2, y1), (0, 0, 255), 1)
            last_point = p
    #             new_image = cv2.circle(new_image, (p[0],p[1]), round(thickness[p[1]][p[0]]), (0,0,255), -1)

    return new_image


def remove_pixels_from_skeleton(img, skeleton):
    return np.logical_and(img, skeleton).astype(np.uint8)


def plot_image(img, title=''):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def plot_image_with_points(img, points):
    _im = img.copy()
    for p in points:
        _im = cv2.circle(_im, (p[0], p[1]), 5, (0, 0, 222), -1)
    plot_image(_im)


def add_points_to_image(img, points, color=(0, 0, 222), add_change=False):
    _im = img.copy()
    for a in points:
        p = a['p']
        change = a['change']
        _im = cv2.circle(_im, (p[0], p[1]), 5, color, -1)

        if add_change:
            _im = cv2.putText(_im, f"{int(change)}%", (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 1, cv2.LINE_AA)
    return _im


def plot_skeletons(img, title):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axs[0, 0].imshow(get_skeleton(img))
    axs[0, 0].set_title('Skeleton')

    axs[0, 1].imshow(get_skeleton(dialate_erode(img)))
    axs[0, 1].set_title('Skeleton with dialate and erode')

    axs[0, 2].imshow(prune(get_skeleton(img)))
    axs[0, 2].set_title('Skeleton with pruned')

    axs[0, 3].imshow(prune(get_skeleton(dialate_erode(img))))
    axs[0, 3].set_title('Skeleton with dialte and erode then pruned')

    axs[1, 0].imshow(np.logical_xor(get_skeleton(img), img))

    axs[1, 1].imshow(np.logical_xor(get_skeleton(dialate_erode(img)), img))

    axs[1, 2].imshow(np.logical_xor(prune(get_skeleton(img)), img))

    axs[1, 3].imshow(np.logical_xor(prune(get_skeleton(dialate_erode(img))), img))

    fig.suptitle(title)
    plt.show()


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
