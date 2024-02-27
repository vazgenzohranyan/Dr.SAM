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
from src.anomality_detection.utils import get_anomality_centers_from_image, add_points_to_image
from src.anomality_detection.skeletonize import skeletonize_v2
from src.anomality_detection.detection_algorithms import find_anomality_points


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
    predictor = SAM_with_many_iter
    args = parse_args()
    data_path = Path(args.data).resolve()

    boxes_df = pd.read_csv(data_path / 'bounding_boxes.csv')
    image_ids = sorted(boxes_df['Image number'].unique())

    output_dir = Path('tmp/').resolve()

    if not output_dir.exists():
        output_dir.mkdir()

    if not (output_dir / 'full').exists():
        (output_dir / 'full').mkdir()

    output_dir = output_dir / 'full'
    print('Starting pipeline')

    for im_id in tqdm(image_ids):
        if not (data_path / f"images/{im_id}.jpg").exists():
            print(f'Image not found with id {im_id}')
            continue

        if not (data_path / f"masks/{im_id}.png").exists():
            print(f'Mask not found with id {im_id}')
            continue

        image = cv2.imread(str(data_path / f"images/{im_id}.jpg"))
        image_with_anomalities = cv2.imread(str(data_path / f"anomality_marked/{im_id}.jpg"))
        anomality_points = get_anomality_centers_from_image(image_with_anomalities)

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

        try:
            predicted_masks = predictor(image, input_boxes)
        except Exception as e:
            print(f"Error with predictor for image {im_id}: {e}")
            continue

        _an_points = [i for p in list(anomality_points.values()) for i in p]
        _image = image.copy()
        _image = add_points_to_image(_image, [{'p': (p[1], p[0]), 'change': 0}
                                              for p in _an_points])

        for j in range(0, 3, 1):
            x1, y1, x2, y2 = bboxes[j]
            predicted_mask = np.transpose(predicted_masks[j].cpu().numpy(), (1, 2, 0))[y1:y2, x1:x2]

            if np.sum(predicted_mask) == 0:
                print(f"Mask {j} is empty for image {im_id}")
                continue

            if np.sum(predicted_mask) is np.nan:
                print(f"Mask {j} is nan for image {im_id}")
                continue

            try:
                # Taking largest areas from binary masks within the bounding box
                predicted_mask = take_largest_contour(predicted_mask)
            except:
                # If not enough data to take the largest contour, take all box
                print(
                    f'Not enough data to take largest contour for image {im_id},box {j}')
            predicted_mask = np.squeeze(predicted_mask)

            skeleton, segmented, segments, _ = skeletonize_v2(predicted_mask)
            pred_an_points, _ = find_anomality_points(predicted_mask)
            _image_with_all_anomalies = add_points_to_image(_image[y1:y2, x1:x2].copy(),
                                                            pred_an_points,
                                                            color=(0, 128, 0))

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(_image[y1:y2, x1:x2], cmap='gray')
            plt.title('Ground truth with anomalies', fontsize=8)

            plt.subplot(1, 4, 2)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title('Predicted mask', fontsize=8)

            plt.subplot(1, 4, 3)
            plt.imshow(np.logical_xor(skeleton, predicted_mask), cmap='gray')
            plt.title('Mask with skeleton', fontsize=8)

            plt.subplot(1, 4, 4)
            plt.imshow(_image_with_all_anomalies, cmap='gray')
            plt.title('Ground truth and predicted anomalies', fontsize=8)

            plt.savefig(output_dir / f'{im_id}_{j}.png')


if __name__ == '__main__':
    main()
