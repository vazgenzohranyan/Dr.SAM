import cv2
import torch
import numpy as np


def take_largest_contour(mask):
    gt = np.squeeze(mask.copy()).astype(np.uint8)
    cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(gt.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(gt, out)[..., None]


def local_adaptive_histogram_equalization(image, grid_size):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
    enhanced_image = clahe.apply(image)
    return enhanced_image


def process_image_with_radius_smoothing(main_img: np.ndarray, guidance_img: np.ndarray, original_img: np.ndarray, radius: int, blur_strength: int) -> np.ndarray:
    """
    Apply smoothing to an image, taking pixels from the original image if there are no surrounding pixels
    with a value of 255 within a specified radius in the guidance image, and from the main image otherwise.

    :param main_img: The main image to process.
    :param guidance_img: The guidance image where the presence of 255-value pixels within a radius affects processing.
    :param original_img: The original image to take pixel values from if conditions are met.
    :param radius: The radius within which to look for 255-value pixels in the guidance image.
    :param blur_strength: The strength of the blur to apply.
    :return: The processed image.
    """
    # Check if images are loaded properly and have the same dimensions
    if main_img is None or guidance_img is None or original_img is None:
        raise ValueError("One or more images are not loaded properly.")
    if main_img.shape != guidance_img.shape or original_img.shape != main_img.shape:
        raise ValueError("All images must have the same dimensions.")

    # Create a copy of the main image to apply changes
    processed_img = np.copy(main_img)

    # Iterate over each pixel in the image
    for i in range(main_img.shape[0]):
        for j in range(main_img.shape[1]):
            # Define the region to check in the guidance image
            x_start = max(0, i - radius)
            x_end = min(guidance_img.shape[0], i + radius + 1)
            y_start = max(0, j - radius)
            y_end = min(guidance_img.shape[1], j + radius + 1)
            region = guidance_img[x_start:x_end, y_start:y_end]

            # Check if any pixel in the region of the guidance image is 255
            if not np.any(region == 255):
                # Take the pixel value from the original image
                processed_img[i, j] = original_img[i, j]
            else:
              pass
                # Optionally, you can apply smoothing to the main image pixels here
                # For example, you could blend the pixel value with a blurred version of the main image
                # This is commented out because it depends on your specific requirements
                # processed_img[i, j] = cv2.GaussianBlur(main_img[i:i+1, j:j+1], (blur_strength, blur_strength), 0)

    return processed_img


def enhancement_short_line_removal(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  enhanced_image_16 = local_adaptive_histogram_equalization(gray, grid_size=16)
  _, binary_image_16 = cv2.threshold(enhanced_image_16, 130, 255, cv2.THRESH_BINARY)


  blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

  edges_original = cv2.Canny(blurred_image, 50, 150)
  # eddges_original = edges_original

  length_threshold = 110
  # Find contours in the edged image
  contours, _ = cv2.findContours(edges_original, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours by length
  long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > length_threshold]

  # Create an empty image to draw the contours
  output = np.zeros_like(edges_original)
  output = cv2.drawContours(output, long_contours, -1, (255, 255, 255), thickness=1)

  processed = process_image_with_radius_smoothing(enhanced_image_16, output, gray, 5, 21)

  return processed


def simple_enhancement(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  enhanced_image = local_adaptive_histogram_equalization(gray, grid_size=2)

  return enhanced_image


# Function to add an extra dimension to the first tensor to match the shape of the second tensor
def add_dimension_to_tensor(tensor, target_tensor):
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


