import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageFilter

from .utils import enhancement_short_line_removal, simple_enhancement
from .point_finders import find_positive_points, find_additional_positive_points, find_min_max_points, \
    find_density_max_point, find_additional_positive_points_near_bbox, find_positive_points_sampled


PREDICTOR = None
DEVICE = 'mps'


def load_model():
    global PREDICTOR
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)

    PREDICTOR = SamPredictor(sam)

load_model()


def create_masks(mask, bounding_boxes):
    """
    Creates a tensor of grayscale extracted regions for each bounding box in an image.
    Each region contains pixel values from the bounding box in the original image,
    with zeros elsewhere.

    :param image_path: Path to the input image file (str).
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


def SAM_with_boxes(image, input_boxes):
    PREDICTOR.set_image(image)
    transformed_boxes = PREDICTOR.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = PREDICTOR.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


def SAM_enhanced_short_line_removal_with_boxes(image, input_boxes):
    enhanced = enhancement_short_line_removal(image)
    if len(enhanced.shape) == 2:
        enhanced = np.stack((enhanced,) * 3, axis=-1)
    else:
        raise ValueError("The image is not a grayscale image.")
    PREDICTOR.set_image(enhanced)
    transformed_boxes = PREDICTOR.transform.apply_boxes_torch(input_boxes, enhanced.shape[:2])
    masks, _, _ = PREDICTOR.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


def SAM_with_boxes_and_points(image, input_boxes, input_point, input_label):
    PREDICTOR.set_image(image)

    transformed_boxes = PREDICTOR.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks = []
    for i in range(0, 3, 1):
        mask, _, _ = PREDICTOR.predict(
            point_coords=input_point[2 * i:2 * i + 2],
            point_labels=input_label[2 * i:2 * i + 2],
            box=transformed_boxes[i].cpu().numpy(),
            multimask_output=False,
        )
        masks.append(mask)
    # Convert to a PyTorch tensor
    tensor_from_numpy = torch.from_numpy(np.array(masks))

    # Move the tensor to CUDA device
    masks = tensor_from_numpy.to(DEVICE)
    return masks


def SAM_with_boxes_and_positive_points(image, input_boxes, input_point, input_label):
    PREDICTOR.set_image(image)

    # transformed_boxes = PREDICTOR.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks = []
    for i in range(0, 3, 1):
        mask, _, _ = PREDICTOR.predict(
            point_coords=np.array([input_point[i], input_point[i + 3]]),
            point_labels=np.array([input_label[i], input_label[i + 3]]),
            box=input_boxes[i].cpu().numpy(),
            multimask_output=False,
        )
        masks.append(mask)
    # Convert to a PyTorch tensor
    tensor_from_numpy = torch.from_numpy(np.array(masks))

    # Move the tensor to CUDA device
    masks = tensor_from_numpy.to(DEVICE)
    return masks

# Here goes different ways of segmentation


def SAM_on_original_image(image, input_boxes):
    return SAM_with_boxes(image, input_boxes)


def SAM_on_enhanced(image, input_boxes):
    enhanced = simple_enhancement(image)
    if len(enhanced.shape) == 2:
        enhanced = np.stack((enhanced,) * 3, axis=-1)
    else:
        raise ValueError("The image is not a grayscale image.")

    return SAM_with_boxes(enhanced, input_boxes)


def SAM_pos_enhanced_neg_original(image, input_boxes):
    input_point, input_label = find_min_max_points(image, input_boxes.cpu().numpy())

    return SAM_with_boxes_and_points(image, input_boxes, input_point, input_label)


def SAM_pos_enhanced_neg_original_1_bbox(image, input_boxes):
    def SAM_with_boxes_and_points(image, input_boxes, input_point, input_label):
        PREDICTOR.set_image(image)

        # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []
        for i in range(0, 3, 1):
            mask, _, _ = PREDICTOR.predict(
                point_coords=np.vstack(([input_point[2 * i:2 * i + 2], input_point[i + 6]])),
                point_labels=np.hstack(([input_label[2 * i:2 * i + 2], input_label[i + 6]])),
                box=input_boxes[i].cpu().numpy(),
                multimask_output=False,
            )
            masks.append(mask)
        # Convert to a PyTorch tensor
        tensor_from_numpy = torch.from_numpy(np.array(masks))

        # Move the tensor to CUDA device
        masks = tensor_from_numpy.to(DEVICE)
        return masks

    input_point, input_label = find_min_max_points(image, input_boxes.cpu().numpy())
    input_point, input_label = find_additional_positive_points_near_bbox(image, input_boxes, input_point,
                                                                         input_label, radius=50, inner_margin=15)

    return SAM_with_boxes_and_points(image, input_boxes, input_point, input_label)


def SAM_2_positive_points_on_original(image, input_boxes):
    input_point, input_label = find_positive_points(image, input_boxes)
    input_point, input_label = find_additional_positive_points(image, input_boxes, input_point, input_label)

    return SAM_with_boxes_and_positive_points(image, input_boxes, input_point, input_label)


def SAM_1_positive_point_on_original_with_sampling(image, input_boxes):
    input_point, input_label = find_positive_points_sampled(image, input_boxes)

    PREDICTOR.set_image(image)

    # transformed_boxes = PREDICTOR.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks = []
    for i in range(0, 3, 1):
        mask, _, _ = PREDICTOR.predict(
            point_coords=np.array([input_point[i]]),
            point_labels=np.array([input_label[i]]),
            box=input_boxes[i].cpu().numpy(),
            multimask_output=False,
        )
        masks.append(mask)
    # Convert to a PyTorch tensor
    tensor_from_numpy = torch.from_numpy(np.array(masks))

    # Move the tensor to CUDA device
    return tensor_from_numpy.to(DEVICE)


def SAM_1_positive_point_1_bbox(image, input_boxes):
    input_point, input_label = find_positive_points(image, input_boxes)
    input_point, input_label = find_additional_positive_points_near_bbox(image, input_boxes, input_point,
                                                                         input_label, radius=50, inner_margin=15)

    return SAM_with_boxes_and_positive_points(image, input_boxes, input_point, input_label)


def SAM_with_many_iter(image, input_boxes, i=3):
    def _predict(image, input_boxes, input_points):
        PREDICTOR.set_image(image)
        #     print(len(input_points))
        # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []
        for box, points in zip(input_boxes, input_points):
            mask, _, _ = PREDICTOR.predict(
                # point_coords=input_point[2*i:2*i+2],
                # point_labels=input_label[2*i:2*i+2],
                point_coords=points,
                point_labels=np.ones(len(points)),
                box=box.cpu().numpy(),
                multimask_output=False,
            )
            masks.append(mask)
            # Convert to a PyTorch tensor
        tensor_from_numpy = torch.from_numpy(np.array(masks))

        # Move the tensor to CUDA device
        masks = tensor_from_numpy.to(DEVICE)
        return masks

    _input_points = []

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


def SAM_with_many_iter_from_sampled(image, input_boxes, i=3):
    def _predict(image, input_boxes, input_points):
        PREDICTOR.set_image(image)
        #     print(len(input_points))
        # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []
        for box, points in zip(input_boxes, input_points):
            mask, _, _ = PREDICTOR.predict(
                # point_coords=input_point[2*i:2*i+2],
                # point_labels=input_label[2*i:2*i+2],
                point_coords=points,
                point_labels=np.ones(len(points)),
                box=box.cpu().numpy(),
                multimask_output=False,
            )
            masks.append(mask)
            # Convert to a PyTorch tensor
        tensor_from_numpy = torch.from_numpy(np.array(masks))

        # Move the tensor to CUDA device
        masks = tensor_from_numpy.to(DEVICE)
        return masks

    _input_points = []

    _input_points = []
    input_points, input_label = find_positive_points_sampled(image, input_boxes)
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

            input_point_v2, input_label_v2 = find_positive_points_sampled(_img, [input_boxes[j]])

            mask_points_v2.extend(input_point_v2)

            input_points_v2[j].extend(mask_points_v2)

    return predicted