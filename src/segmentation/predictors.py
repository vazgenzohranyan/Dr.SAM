import torch
import numpy as np
from PIL import Image, ImageFilter
from segment_anything import sam_model_registry, SamPredictor

from .point_finders import find_positive_points, find_additional_positive_points


PREDICTOR = None
DEVICE = 'mps'  # Change this if you are running on CUDA or CPU


def load_model(checkpoint_path: str = "sam_vit_h_4b8939.pth",
               model_type: str = "vit_h",
               device: str = 'mps'):
    """
    Loads model to the global variable
    :return:
    """
    global PREDICTOR, DEVICE

    if PREDICTOR is not None:
        return PREDICTOR

    DEVICE = device

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)

    PREDICTOR = SamPredictor(sam)

    return PREDICTOR


def create_masks(mask: torch.Tensor, bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    Creates a tensor of grayscale extracted regions for each bounding box in an image.
    Each region contains pixel values from the bounding box in the original image,
    with zeros elsewhere.

    :param mask: Binary mask of the image (torch.Tensor).
    :param bounding_boxes: Tensor of bounding boxes, each row is [x_min, y_min, x_max, y_max] (torch.Tensor).
    :return: A tensor containing grayscale extracted regions for each bounding box.
    """
    # Load the image, convert to grayscale and then to a numpy array
    image_np = mask
    height, width = image_np.shape

    # Convert the image to a tensor
    image_tensor = torch.from_numpy(image_np).float().to(device=DEVICE)

    # Initialize a list to hold the individual region tensors
    regions = []

    # Extract regions for each bounding box
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        # Initialize a zeroed tensor for the region
        region = torch.zeros((height, width), device=DEVICE)

        # Copy the pixels from the bounding box region of the original image
        region[y_min:y_max, x_min:x_max] = image_tensor[y_min:y_max, x_min:x_max]

        # Append the region tensor to the list
        regions.append(region)

    # Combine the individual regions into a single tensor
    combined_regions = torch.stack(regions)

    return combined_regions


def segmentize(image: np.ndarray, input_boxes: torch.Tensor, i: int = 3) -> torch.Tensor:
    """
    Segments the image using Dr.SAM algorithm.
    :param image: Grayscale image to segment.
    :param input_boxes: Torch tensor representing the bounding boxes of the image to segment.
    :param i: Hyperparameter for the number of iterations to run the algorithm after the main part.
    :return:
    """
    def _predict(image, input_boxes, input_points):
        PREDICTOR.set_image(image)

        masks = []
        for box, points in zip(input_boxes, input_points):
            mask, _, _ = PREDICTOR.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                box=box.cpu().numpy(),
                multimask_output=False,
            )
            masks.append(mask)
            # Convert to a PyTorch tensor
        tensor_from_numpy = torch.from_numpy(np.array(masks))

        # Move the tensor to device
        masks = tensor_from_numpy.to(DEVICE)
        return masks

    predicted = None
    _input_points = []

    input_points, input_label = find_positive_points(image, input_boxes)
    input_points, input_label = find_additional_positive_points(image, input_boxes, input_points, input_label)
    input_points_v2 = [[input_points[i], input_points[i + 3]] for i in range(3)]

    for _ in range(i):
        predicted = _predict(image, input_boxes, np.array(input_points_v2))
        for j in range(0, 3, 1):
            mask_points_v2 = []
            mask1 = np.transpose(predicted[j].cpu().numpy(), (1, 2, 0))

            im_pil = Image.fromarray(np.squeeze(mask1))
            im_pil = im_pil.filter(ImageFilter.ModeFilter(size=7))
            fixed_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))

            _img = image.copy()
            _img[fixed_mask.astype(np.bool8)] = [255, 255, 255]

            input_point_v2, input_label_v2 = find_positive_points(_img, [input_boxes[j]])

            mask_points_v2.extend(input_point_v2)

            input_points_v2[j].extend(mask_points_v2)

    return predicted
