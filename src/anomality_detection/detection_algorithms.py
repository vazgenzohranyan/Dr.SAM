import numpy as np
from sklearn.cluster import DBSCAN

from .skeletonize import skeletonize, get_thickness_from_distance_transform, get_thickness_from_skeleton


def find_anomality_points(mask: np.ndarray) -> tuple[list, list]:
    """
    Finds anomaly points on the mask using skeletonization algorithm.
    :param mask: Numpy array N x M representing the mask.
    :return: Tuple containing the points(list with 2D indexes) and the anomaly indicator(representing anomaly existence
             in the branches of the skeleton).
    """
    test_image = mask.copy()
    skeleton, segmented_skeleton, segments, r = skeletonize(test_image)
    thickness = get_thickness_from_skeleton(test_image)
    thickness_dt = get_thickness_from_distance_transform(test_image, skeleton)

    points = []
    anomality_indicator = [False] * len(segments)
    _im = test_image.copy()

    for _, seg in enumerate(segments):
        thicknesses = []
        thicknesses_dt = []
        for [p] in seg[:len(seg) // 2]:
            _im[p[1]][p[0]] = 1
            thicknesses.append(thickness[p[1]][p[0]])
            thicknesses_dt.append(thickness_dt[p[1]][p[0]])

        # Removing some pixels from the start and end of the thicknesses array as they are not reliable

        if _ == len(segments) - 1:
            _thickness = np.array(thicknesses[40:-40])
            _thickness_dt = np.array(thicknesses_dt[40:-40])
            hop = 40
        elif len(thicknesses) > 50:
            _thickness = np.array(thicknesses[20:-20])
            _thickness_dt = np.array(thicknesses_dt[20:-20])
            hop = 20
        elif len(thicknesses) > 30:
            _thickness = np.array(thicknesses[10:-10])
            _thickness_dt = np.array(thicknesses_dt[10:-10])
            hop = 10
        else:
            _thickness = np.array(thicknesses[5:-5])
            _thickness_dt = np.array(thicknesses_dt[5:-5])
            hop = 5

        _mean_dt = (_thickness + _thickness_dt) / 2
        anomalies = []

        # Finding extreme points in the thickness array

        for i, p in enumerate(_thickness):
            anomaly = True
            for j in range(10):
                if i - j >= 0:
                    left_side = _thickness[i - j]
                else:
                    left_side = p

                if i + j < len(_thickness):
                    right_side = _thickness[i + j]
                else:
                    left_side = p

                if left_side < p < right_side or left_side > p > right_side:
                    anomaly = False

            if anomaly:
                anomalies.append(i)

        real_anomalies = _thickness[anomalies]
        anomaly_indices = np.array(anomalies)

        temp = []
        filtered_anomalies = []
        i = 0

        # Removing found extreme points laying on the same line

        while i < len(real_anomalies):
            temp.append(i)

            for j in range(i + 1, len(real_anomalies)):
                if real_anomalies[j] == real_anomalies[temp[-1]]:
                    temp.append(j)
                else:
                    filtered_anomalies.append([np.mean(anomaly_indices[temp]), real_anomalies[temp[-1]]])
                    temp = []
                    break

            if j == (len(real_anomalies) - 1):
                if len(temp) > 0:
                    filtered_anomalies.append([np.mean(anomaly_indices[temp]), real_anomalies[temp[-1]]])
                break

            i = j
        filtered_anomalies = np.array(filtered_anomalies)

        if len(filtered_anomalies) > 0:
            ff_anomalies = [filtered_anomalies[0]]

            # Removing anomalies laying on the slopes

            for i, a in enumerate(filtered_anomalies[1:-1]):
                if (ff_anomalies[-1][1] < a[1] > filtered_anomalies[i + 2][1] or
                        ff_anomalies[-1][1] > a[1] < filtered_anomalies[i + 2][1]):
                    ff_anomalies.append(a)
            ff_anomalies.append(filtered_anomalies[-1])
            filtered_anomalies = np.array(ff_anomalies)

            # Clustering the anomalies to remove the noise

            clustering = DBSCAN(eps=len(_thickness) / 10, min_samples=1).fit(filtered_anomalies)

            clustered_anomalies = []
            for c in np.unique(clustering.labels_):
                indices = np.where(clustering.labels_ == c)[0]
                if len(indices) > 1:
                    group_anomalies = [a[0] for a in filtered_anomalies[indices]]
                    _ind = int(np.mean(group_anomalies))
                    clustered_anomalies.append((_ind, _thickness[_ind]))
                else:
                    clustered_anomalies.append(filtered_anomalies[indices][0])
            filtered_anomalies = np.array(clustered_anomalies)

            ff_anomalies = [filtered_anomalies[0]]

            for i, a in enumerate(filtered_anomalies[1:-1]):
                if (ff_anomalies[-1][1] < a[1] > filtered_anomalies[i + 2][1] or
                        ff_anomalies[-1][1] > a[1] < filtered_anomalies[i + 2][1]):
                    ff_anomalies.append(a)
            ff_anomalies.append(filtered_anomalies[-1])
            filtered_anomalies = np.array(ff_anomalies)

            changes = []
            changes_to_show = []
            if len(filtered_anomalies) > 1:
                # Calculating the change in thickness and distance transform values

                for i, a in enumerate(filtered_anomalies):
                    d = a[1]  # a[1] for thinning
                    _left = max(0, int(a[0] - len(_thickness) / 5))
                    _right = min(len(_thickness) - 1, int(a[0] + len(_thickness) / 5))

                    _mean = (int(_thickness[_left]) + int(_thickness[_right])) / 2
                    _mean_dt = (int(_thickness_dt[_left]) + int(_thickness_dt[_right])) / 2

                    if _mean == 0:
                        changes.append(0)
                    else:
                        _change = (abs(d - _mean) / _mean) * 100
                        _direction = 1 if d > _mean else -1
                        changes.append(_change * _direction)

                    if _mean_dt == 0:
                        changes_to_show.append(0)
                    else:
                        _change = (abs(_thickness_dt[int(a[0])] - _mean_dt) / _mean_dt) * 100
                        _direction = 1 if d > _mean_dt else -1
                        changes_to_show.append(_change * _direction)
            else:
                changes = [0]
            changes = np.array(changes)
            changes_to_show = np.array(changes_to_show)

            # Filtering the anomalies based on the change values

            anomality_arr = (np.abs(changes) >= 40)

            filtered_anomalies = filtered_anomalies[anomality_arr]
            changes_to_show = changes_to_show[anomality_arr]

            if len(filtered_anomalies) > 0:
                anomality_indicator[_] = True

            for a, change in zip(filtered_anomalies, changes_to_show):
                points.append({'p': seg[hop + int(a[0])][0], 'change': change})

    return points, anomality_indicator