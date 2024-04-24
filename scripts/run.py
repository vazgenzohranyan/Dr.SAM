import sys
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

sys.path.insert(1, str(Path(__file__).parent.parent.resolve()))

from src.segmentation.utils import take_largest_contour
from src.segmentation.predictors import DEVICE, segmentize, load_model
from src.anomality_detection.detection_algorithms import find_anomality_points
from src.anomality_detection.utils import add_points_to_image, draw_circles_from_segments
from src.anomality_detection.skeletonize import skeletonize, get_thickness_from_distance_transform


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        required=True,
        type=str,
        help='Description of argument.'
    )
    parser.add_argument(
        '--images',
        nargs='+',
        type=int,
        help='List of image ids',
        required=False
    )
    parser.add_argument(
        '--only-segment',
        help='Run only segmentation part',
        action='store_true',
        default=False
    )
    return parser.parse_args(*argument_array)


def main():
    load_model()

    args = parse_args()

    data_path = Path(args.data).resolve()

    metadata_path = data_path / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found in {data_path}")

    metadata_df = pd.read_json(metadata_path)

    image_ids = sorted(metadata_df['image_id'].unique())

    output_dir = Path('tmp/').resolve()

    if not output_dir.exists():
        output_dir.mkdir()

    print('Starting pipeline')

    if args.images and len(args.images) > 0:
        ids = args.images
    else:
        ids = image_ids

    for im_id in tqdm(ids):
        if not (data_path / f"images/{im_id}.jpg").exists():
            print(f'Image not found with id {im_id}')
            continue

        if not (data_path / f"masks/{im_id}.png").exists():
            print(f'Mask not found with id {im_id}')
            continue

        image = cv2.imread(str(data_path / f"images/{im_id}.jpg"))
        metadata = metadata_df[metadata_df['image_id'] == im_id].iloc[0].to_dict()

        mask = cv2.imread(str(data_path / f"masks/{im_id}.png"))
        mask = 255 - cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        if np.sum(mask) == 0:
            print(f'Mask is empty for image {im_id}')
            continue

        bboxes = np.array(metadata['bboxes'])
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]

        input_boxes = torch.from_numpy(bboxes).to(device=DEVICE)

        try:
            predicted_masks = segmentize(image, input_boxes)
        except Exception as e:
            print(f"Error with predictor for image {im_id}: {e}")
            continue

        _an_points = [i for p in list(metadata['anomalies'].values()) for i in p]
        _image = image.copy()
        _image = add_points_to_image(_image, [{'p': (p[0], p[1]), 'change': 0}
                                              for p in _an_points])

        for j in range(0, 3, 1):
            x1, y1, x2, y2 = bboxes[j]

            try:
                predicted_mask = np.transpose(predicted_masks[j].cpu().numpy(), (1, 2, 0))[y1:y2, x1:x2]
            except:
                predicted_mask = np.transpose(predicted_masks[j], (1, 2, 0))[y1:y2, x1:x2]

            if np.sum(predicted_mask) == 0:
                print(f"Mask {j} is empty for image {im_id}")
                continue

            if np.sum(predicted_mask) is np.nan:
                print(f"Mask {j} is nan for image {im_id}")
                continue

            try:
                # Taking largest areas from binary masks within the bounding box
                predicted_mask = take_largest_contour(predicted_mask)
            except Exception as e:
                # If not enough data to take the largest contour, take all box
                print(
                    f'Not enough data to take largest contour for image {im_id},box {j}')

            predicted_mask = np.squeeze(predicted_mask)
            im_pil = Image.fromarray(predicted_mask)
            fixed_predicted_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))

            if not (output_dir / f'{im_id}').exists():
                (output_dir / f'{im_id}').mkdir()

            plt.figure(figsize=(15, 8))
            plt.imshow(predicted_mask, cmap='gray')
            plt.yticks([])
            plt.xticks([])
            plt.savefig(output_dir / f'{im_id}' / f'{j+1}_predicted_mask.png',
                        transparent=True, bbox_inches='tight', dpi=200)
            plt.close()

            if args.only_segment:
                continue

            skeleton, segmented, segments, _ = skeletonize(fixed_predicted_mask)
            pred_an_points, _ = find_anomality_points(fixed_predicted_mask)

            _image_with_all_anomalies = add_points_to_image(_image[y1:y2, x1:x2].copy(),
                                                            pred_an_points,
                                                            add_change=True,
                                                            color=(105,0,0))

            thickness = get_thickness_from_distance_transform(fixed_predicted_mask, skeleton)

            img_with_circles = draw_circles_from_segments(fixed_predicted_mask,
                                                          thickness=thickness,
                                                          segments=segments)

            plt.imshow(np.logical_xor(skeleton, fixed_predicted_mask), cmap='gray')
            plt.yticks([])
            plt.xticks([])
            plt.savefig(output_dir / f'{im_id}' / f'{j+1}_skeleton.png',
                        transparent=True, bbox_inches='tight', dpi=200)
            plt.close()

            plt.imshow(img_with_circles, cmap='gray')
            plt.yticks([])
            plt.xticks([])
            plt.savefig(output_dir / f'{im_id}' / f'{j+1}_diameters.png',
                        transparent=True, bbox_inches='tight', dpi=200)
            plt.close()

            plt.imshow(_image_with_all_anomalies, cmap='gray')
            plt.yticks([])
            plt.xticks([])
            plt.savefig(output_dir / f'{im_id}' / f'{j+1}_anomalies.png',
                        transparent=True, bbox_inches='tight', dpi=200)
            plt.close()


if __name__ == '__main__':
    main()
