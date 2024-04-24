import cv2
import torch
import numpy as np


def take_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Take the largest contour from a binary mask.
    :param mask: A numpy array representing a binary mask.
    :return: A numpy array representing the largest contour.
    """
    gt = np.squeeze(mask.copy()).astype(np.uint8)
    cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(gt.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(gt, out)[..., None]


def add_dimension_to_tensor(tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Add an extra dimension to a tensor to match the shape of target tensor.
    :param tensor: The tensor to add an extra dimension to.
    :param target_tensor: The target tensor with the desired shape.
    :return: The tensor with the added dimension.
    """
    # Get the number of dimensions of the target tensor
    target_dims = len(target_tensor.shape)

    # Expand the tensor to match the dimensions of the target tensor
    while len(tensor.shape) < target_dims:
        tensor = tensor.unsqueeze(1)

    return tensor


