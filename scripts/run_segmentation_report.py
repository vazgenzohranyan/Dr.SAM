import json
import sys
import cv2
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent.parent.resolve()))

from src.segmentation.predictors import PREDICTOR, create_masks, SAM_on_original_image, \
    SAM_on_enhanced, SAM_pos_enhanced_neg_original, SAM_pos_enhanced_neg_original_1_bbox, \
    SAM_2_positive_points_on_original, SAM_1_positive_point_on_original_with_sampling, \
    SAM_1_positive_point_1_bbox, SAM_with_many_iter, SAM_with_many_iter_from_sampled
from src.segmentation.utils import take_largest_contour, add_dimension_to_tensor

AVAILABLE_PREDICTORS = [
    # {
    #     'description': 'SAM on original images',
    #     'predictor': SAM_on_original_image
    # },
    # {
    #     'description': 'SAM on enhanced images',
    #     'predictor': SAM_on_enhanced
    # },
    # {
    #     'description': 'SAM, input points (positives from the enhanced image, negatives from the original image) based on values',
    #     'predictor': SAM_pos_enhanced_neg_original
    # },
    # {
    #     'description': 'SAM, input points (positives from the enhanced image, negatives from the original image) based on values + 1 near bbox',
    #     'predictor': SAM_pos_enhanced_neg_original_1_bbox
    # },
    # {
    #     'description': 'SAM 2 positive points on original image',
    #     'predictor': SAM_2_positive_points_on_original
    # },
    # {
    #     'description': 'SAM 1 positive points on original image with sampling',
    #     'predictor': SAM_1_positive_point_on_original_with_sampling
    # },
    # {
    #     'description': 'SAM 1 positive points on original image plus 1 near bbox',
    #     'predictor': SAM_1_positive_point_1_bbox
    # },
    # {
    #     'description': 'SAM 2 positive points plus additional points with iteration',
    #     'predictor': SAM_with_many_iter
    # },
    # {
    #     'description': 'SAM 2 positive points plus additional points with iteration (with sampling)',
    #     'predictor': SAM_with_many_iter_from_sampled
    # }
]


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        required=True,
        type=str,
        help='Description of argument.'
    )

    return parser.parse_args(*argument_array)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()

    boxes_df = pd.read_csv(data_path / 'bounding_boxes.csv')
    image_ids = sorted(boxes_df['Image number'].unique())

    ious = [
        {
            'description': p['description'],
            'ious': []
        }
        for p in AVAILABLE_PREDICTORS
    ]

    print('Starting segmentation')

    for im_id in tqdm(image_ids):
        if not (data_path / f"images/{im_id}.jpg").exists():
            print(f'Image not found with id {im_id}')
            continue

        if not (data_path / f"masks/{im_id}.png").exists():
            print(f'Mask not found with id {im_id}')
            continue

        image = cv2.imread(str(data_path / f"images/{im_id}.jpg"))
        mask = cv2.imread(str(data_path / f"masks/{im_id}.png"))
        mask = 255 - cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        if np.sum(mask) == 0:
            print(f'Mask is empty for image {im_id}')
            continue

        bboxes = boxes_df[boxes_df['Image number'] == im_id][['x', 'y', 'width', 'height']]
        bboxes['width'] = bboxes['x'] + bboxes['width']
        bboxes['height'] = bboxes['y'] + bboxes['height']
        bboxes = bboxes.to_numpy()
        input_boxes = torch.from_numpy(bboxes).to(device=PREDICTOR.device)

        for i, p in enumerate(AVAILABLE_PREDICTORS):
            try:
                predicted_masks = p['predictor'](image, input_boxes)
            except Exception as e:
                print(f"Error with predictor {p['description']} for image {im_id}: {e}")
                continue

            ground_truth_masks = create_masks(mask, input_boxes)
            ground_truth_masks = add_dimension_to_tensor(ground_truth_masks, predicted_masks)

            for j in range(0, 3, 1):
                x1, y1, x2, y2 = bboxes[j]
                mask1 = np.transpose(predicted_masks[j].cpu().numpy(), (1, 2, 0))[y1:y2, x1:x2]
                mask2 = np.transpose(ground_truth_masks[j].cpu().numpy(), (1, 2, 0))[y1:y2, x1:x2]

                if np.sum(mask1) == 0 or np.sum(mask2) == 0:
                    print(f"Mask {j} is empty for image {im_id} and predictor {p['description']}")
                    continue

                if np.sum(mask1) is np.nan:
                    print(f"Mask {j} is nan for image {im_id} and predictor {p['description']}")
                    continue

                try:
                    # Taking largest areas from binary masks within the bounding box
                    mask1 = take_largest_contour(mask1)
                    mask2 = take_largest_contour(mask2)
                except:
                    # If not enough data to take the largest contour, take all box
                    print(
                        f'Not enough data to take largest contour for image {im_id},box {j} and predictor {p["description"]}')

                # Calculate Intersection and Union
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                iou = np.sum(intersection) / np.sum(union)

                ious[i]['ious'].append(iou)

    for iou in ious:
        print("---------------------------------------------")
        print(
            f"{iou['description']}: Mean - {np.mean(iou['ious'])}, min - {np.min(iou['ious'])}, max - {np.max(iou['ious'])}")


if __name__ == '__main__':
    main()
