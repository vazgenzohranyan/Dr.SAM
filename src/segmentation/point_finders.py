import cv2
import torch
import random
import numpy as np


def calculate_pixel_probabilities(image: np.ndarray) -> np.ndarray:
    """
    Calculate pixel probabilities based on pixel values in the image.
    :param image: A grayscale image array.
    :return: A 2D array of pixel probabilities.
    """
    # Assuming darker pixels have higher probability
    # Normalize pixel values to range [0, 1] and invert (so darker pixels have higher values)
    probabilities = 1.0 - (image / 255.0)
    return probabilities


def exclude_near_points(probabilities: np.ndarray, points: list, radius: int) -> np.ndarray:
    """
    Exclude points within a certain radius of existing points by drawing filled circles around them.
    :param probabilities: Probability map of the image.
    :param points: Points to exclude.
    :param radius: Radius of the filled circle (exclude radius).
    :return: A 2D array of pixel probabilities with points excluded.
    """
    for point in points:
        y, x = point
        cv2.circle(probabilities, (x, y), radius, 0, -1)  # Drawing a filled circle with zero probabilities
    return probabilities


def find_positive_points(image: np.ndarray, bounding_boxes: torch.Tensor, radius: int = 75) -> tuple[np.ndarray, np.ndarray]:
    """
    Find positive points within the bounding boxes of an image by taking the point with the highest density.
    :param image: Grayscale image array.
    :param bounding_boxes: torch.Tensor of bounding boxes.
    :param radius: Radius to consider for density calculation.
    :return: Tuple of points and labels.
    """
    points = []
    labels = []
    image_tensor = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    probabilities = calculate_pixel_probabilities(image_tensor)

    for box in bounding_boxes:
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box

        roi = probabilities[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Flatten the ROI and get the top 100 pixel indices based on probability
        flat_indices = np.argsort(roi.ravel())[-100:]
        y_indices, x_indices = np.unravel_index(flat_indices, roi.shape)
        sampled_points = list(zip(y_indices, x_indices))

        # Find the point with the highest density
        best_point = find_density_max_point(roi, sampled_points, radius)
        if best_point:
            # Adjusting point to be relative to the original image
            best_point = [x1 + best_point[1], y1 + best_point[0]]
            points.append(best_point)
            labels.append(1)
    return np.array(points), np.array(labels)


def find_additional_positive_points(image: np.ndarray,
                                    bounding_boxes: torch.Tensor,
                                    existing_points: np.ndarray,
                                    existing_labels: np.ndarray,
                                    radius: int = 50,
                                    exclusion_radius: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds next positive points within the bounding boxes of an image by taking the point with the highest density
    excluding regions around existing points.
    :param image: Grayscale image array.
    :param bounding_boxes: Bounding boxes of the image.
    :param existing_points: Already existing points.
    :param existing_labels: Already existing labels.
    :param radius: Radius to consider for density calculation.
    :param exclusion_radius: Radius to exclude around existing points.
    :return: Tuple of updated points and labels.
    """
    if isinstance(existing_points, torch.Tensor):
        existing_points = existing_points.cpu().numpy()

    image_tensor = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    probabilities = calculate_pixel_probabilities(image_tensor)

    probabilities = exclude_near_points(probabilities, existing_points, exclusion_radius)

    if isinstance(bounding_boxes, torch.Tensor):
        bounding_boxes = bounding_boxes.cpu().numpy()

    new_points = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box

        roi = probabilities[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        flat_indices = np.argsort(roi.ravel())[-100:]
        y_indices, x_indices = np.unravel_index(flat_indices, roi.shape)
        sampled_points = list(zip(y_indices, x_indices))

        best_point = find_density_max_point(roi, sampled_points, radius)
        if best_point:
            best_point = [x1 + best_point[1], y1 + best_point[0]]
            new_points.append(best_point)
        existing_labels = np.append(existing_labels, 1)

    # Combine existing points with new points
    updated_points = np.vstack((existing_points, new_points))

    return updated_points, existing_labels


def find_min_max_points(image: np.ndarray, bounding_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find a random point with maximum and minimum values for each bounding box in the image.
    :param image: A grayscale or color image array.
    :param bounding_boxes: An iterable of bounding box coordinates, each defined as [x1, y1, x2, y2].
    :return: Tuple of points and labels. Points are an array of [x, y] coordinates and labels are an array of 0s and 1s.
             Label 0 indicates a max value point and label 1 indicates a min value point.
    """
    if len(image.shape) == 3:  # If the image is color, convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    points = np.empty((0, 2), int)
    labels = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Find max value point in ROI
        max_val = np.max(roi)
        max_points = np.argwhere(roi == max_val)
        max_point = random.choice(max_points)  # Randomly select one max point
        max_point_adjusted = [x1 + max_point[1], y1 + max_point[0]]
        points = np.append(points, [max_point_adjusted], axis=0)
        labels.append(0)  # Label for max point

        # Find min value point in ROI
        min_val = np.min(roi)
        min_points = np.argwhere(roi == min_val)
        min_point = random.choice(min_points)  # Randomly select one min point
        min_point_adjusted = [x1 + min_point[1], y1 + min_point[0]]
        points = np.append(points, [min_point_adjusted], axis=0)
        labels.append(1)  # Label for min point

    return points, np.array(labels)


def find_density_max_point(roi: int, sampled_points: list, radius: int) -> list:
    """
    Find the point with the highest density within a radius.
    :param roi: Region of interest.
    :param sampled_points: Sampled points within the ROI.
    :param radius: Radius to consider for density calculation.
    :return: Returns the point with the highest density.
    """
    max_density = 0
    best_point = None
    for point in sampled_points:
        y, x = point
        # Calculate density within radius
        density = sum(np.sqrt((y - sp[0])**2 + (x - sp[1])**2) <= radius for sp in sampled_points)
        if density > max_density:
            max_density = density
            best_point = point
    return best_point


def find_positive_points_sampled(image, bounding_boxes, radius=75):
    """
    Find positive points within the bounding boxes of an image by sampling points and taking the point with the highest density.
    :param image: Grayscale image array.
    :param bounding_boxes: Bounding boxes of the image.
    :param radius: Radius to consider for density calculation.
    :return: Points and labels.
    """
    points = []
    labels = []
    image_tensor = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    probabilities = calculate_pixel_probabilities(image_tensor)

    for box in bounding_boxes:
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box

        roi = probabilities[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Flatten the ROI and get indices of the top 3000 pixel based on probability
        flat_indices = np.argsort(roi.ravel())[-3000:]
        # Randomly sample 100 pixels from these top 3000 pixels
        sampled_indices = random.sample(list(flat_indices), 100)
        y_indices, x_indices = np.unravel_index(sampled_indices, roi.shape)
        sampled_points = list(zip(y_indices, x_indices))

        # Find the point with the highest density
        best_point = find_density_max_point(roi, sampled_points, radius)
        if best_point:
            # Adjusting point to be relative to the original image
            best_point = [x1 + best_point[1], y1 + best_point[0]]
            points.append(best_point)
            labels.append(1)
    return np.array(points), np.array(labels)
