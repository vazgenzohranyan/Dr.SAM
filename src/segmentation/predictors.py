import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageFilter
import time
from .utils import enhancement_short_line_removal, simple_enhancement
from .point_finders import find_positive_points, find_additional_positive_points, find_min_max_points, \
    find_density_max_point, find_additional_positive_points_near_bbox, find_positive_points_sampled

from ..anomality_detection.skeletonize import get_thickness_from_skeleton_v2

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


def SAM_on_equalized_original_image(image, input_boxes):
    image_src = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_eq = cv2.equalizeHist(image_src)
    image = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)
    return SAM_with_boxes(image, input_boxes)

def SAM_on_thresholded_original_image(image, input_boxes):
    _img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    _img = (_img < 210).astype(np.uint8)

    _img = 255 - cv2.cvtColor(_img * 255, cv2.COLOR_GRAY2RGB)

    return SAM_with_boxes(_img, input_boxes)


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


def SAM_2_positive_points_on_equalised_original(image, input_boxes):
    input_point, input_label = find_positive_points(image, input_boxes)
    input_point, input_label = find_additional_positive_points(image, input_boxes, input_point, input_label)

    image_src = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_eq = cv2.equalizeHist(image_src)
    image = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)

    return SAM_with_boxes_and_positive_points(image, input_boxes, input_point, input_label)

def SAM_2_positive_points_mixed(image, input_boxes):
    masks1 = SAM_2_positive_points_on_original(image, input_boxes)
    masks2 = SAM_2_positive_points_on_equalised_original(image, input_boxes)

    masks = []
    for j in range(3):
        mask1 = np.transpose(masks1[j].cpu().numpy(), (1, 2, 0))
        mask2 = np.transpose(masks2[j].cpu().numpy(), (1, 2, 0))
        mask = np.logical_and(mask1, mask2)
        masks.append(np.transpose(mask, (2, 0, 1)))

    return np.array(masks)


def SAM_2_positive_points_on_original_with_thresholded_input(image, input_boxes):
    input_point, input_label = find_positive_points(image, input_boxes)
    input_point, input_label = find_additional_positive_points(image, input_boxes, input_point, input_label)

    _img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    _img = (_img < 210).astype(np.uint8)

    _img = 255 - cv2.cvtColor(_img * 255, cv2.COLOR_GRAY2RGB)

    return SAM_with_boxes_and_positive_points(_img, input_boxes, input_point, input_label)


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

def SAM_1_positive_point_on_original(image, input_boxes):
    input_point, input_label = find_positive_points(image, input_boxes)

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

times = []
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

    start_time = time.time()
    _predict(image, input_boxes, np.array(input_points_v2))
    times.append((time.time() - start_time)/3)
    print("Mean time for segmentation", np.mean(times))

    return predicted


def SAM_with_many_iter_v2(image, input_boxes, i=4):
    def _predict(image, input_boxes, input_points, input_labels):
        print(input_labels, input_points)
        PREDICTOR.set_image(image)
        #     print(len(input_points))
        # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []
        for box, points, labels in zip(input_boxes, input_points, input_labels):
            mask, _, _ = PREDICTOR.predict(
                # point_coords=input_point[2*i:2*i+2],
                # point_labels=input_label[2*i:2*i+2],
                point_coords=np.array(points),
                point_labels=np.array(labels),
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
    input_labels_v2 = [[input_label[i], input_label[i + 3]] for i in range(3)]

    last_input_points = input_points_v2.copy()
    last_input_labels = input_labels_v2.copy()

    for _ in range(i):
        predicted = _predict(image, input_boxes, input_points_v2, input_labels_v2)

        for j in range(0, 3, 1):
            mask_points_v2 = []
            mask1 = np.transpose(predicted[j].cpu().numpy(), (1, 2, 0))

            im_pil = Image.fromarray(np.squeeze(mask1))
            im_pil = im_pil.filter(ImageFilter.ModeFilter(size=7))
            fixed_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))
            distance = cv2.distanceTransform(fixed_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5).astype(
                np.float32)

            max_t = np.max(distance)

            if max_t < 50:
                last_input_points[j] = input_points_v2[j].copy()
                last_input_labels[j] = input_labels_v2[j].copy()
            else:
                if input_labels_v2[j][-1] == -1:
                    continue
                l = np.argwhere(distance==max_t)
                input_points_v2[j].append(l[0])
                input_labels_v2[j].append(-1)
                continue

            _img = image.copy()
            _img[fixed_mask.astype(np.bool8)] = [255, 255, 255]

            input_point_v2, input_label_v2 = find_positive_points(_img, [input_boxes[j]])

            mask_points_v2.extend(input_point_v2)

            input_points_v2[j].extend(mask_points_v2)
            input_labels_v2[j].extend([1]*len(mask_points_v2))

    return _predict(image, input_boxes, last_input_points, last_input_labels)

def SAM_with_many_iter_on_equalised(image, input_boxes, i=3):
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

    image_src = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_eq = cv2.equalizeHist(image_src)
    image_eq = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)

    for _ in range(i):
        predicted = _predict(image_eq, input_boxes, np.array(input_points_v2))
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


def SAM_with_many_iter_on_mixed(image, input_boxes, i=3):
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

    predicted = SAM_2_positive_points_mixed(image, input_boxes)

    image_src = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_eq = cv2.equalizeHist(image_src)
    image_eq = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)

    for _ in range(i):
        for j in range(0, 3, 1):
            mask_points_v2 = []
            mask1 = np.transpose(predicted[j], (1, 2, 0))

            im_pil = Image.fromarray(np.squeeze(mask1))
            im_pil = im_pil.filter(ImageFilter.ModeFilter(size=7))
            fixed_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))

            processed_img = cv2.dilate(fixed_mask.astype(np.uint8), np.ones((17, 17)), 20)

            _img = image.copy()
            _img[fixed_mask.astype(np.bool8)] = [255, 255, 255]
            _img[~processed_img.astype(np.bool8)] = [255, 255, 255]

            input_point_v2, input_label_v2 = find_positive_points(_img, [input_boxes[j]])

            mask_points_v2.extend(input_point_v2)

            input_points_v2[j].extend(mask_points_v2)

        predicted = _predict(image, input_boxes, np.array(input_points_v2))
        _predicted = _predict(image_eq, input_boxes, np.array(input_points_v2))

        masks = []
        for j in range(3):
            mask1 = np.transpose(predicted[j].cpu().numpy(), (1, 2, 0))
            mask2 = np.transpose(_predicted[j].cpu().numpy(), (1, 2, 0))
            mask = np.logical_and(mask1, mask2)
            masks.append(np.transpose(mask, (2, 0, 1)))

        predicted = np.array(masks)

    return predicted


def SAM_with_many_iter_on_original_images(image, input_boxes, i=3):
    def _predict(image, input_boxes, input_points, ind):
        PREDICTOR.set_image(image)
        #     print(len(input_points))
        # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []
        for box, points in zip(input_boxes, input_points):
            if ind:
                mask, _, _ = PREDICTOR.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points)),
                    box=box.cpu().numpy(),
                    multimask_output=False,
                )
            else:
                mask, _, _ = PREDICTOR.predict(
                    box=box.cpu().numpy(),
                    multimask_output=False,
                )
            masks.append(mask)
            # Convert to a PyTorch tensor
        tensor_from_numpy = torch.from_numpy(np.array(masks))

        # Move the tensor to CUDA device
        masks = tensor_from_numpy.to(DEVICE)
        return masks

    input_points_v2 = [[] for i in range(3)]

    for _ in range(i):
        predicted = _predict(image, input_boxes, np.array(input_points_v2), _)
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