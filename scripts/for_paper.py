import json
import sys
import cv2
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

sys.path.insert(1, str(Path(__file__).parent.parent.resolve()))

from src.segmentation.predictors import PREDICTOR, SAM_with_many_iter
from src.segmentation.utils import take_largest_contour
from src.anomality_detection.utils import get_anomality_centers_from_image, add_points_to_image, \
    draw_circles_from_segments
from src.anomality_detection.skeletonize import skeletonize_v2, get_thickness_from_skeleton
from src.anomality_detection.detection_algorithms import find_anomality_points


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
        required=True
    )

    return parser.parse_args(*argument_array)


def main():
    predictor = SAM_with_many_iter
    args = parse_args()
    data_path = Path(args.data).resolve()

    boxes_df = pd.read_csv(data_path / 'bounding_boxes.csv')

    output_dir = Path('tmp/').resolve()

    if not output_dir.exists():
        output_dir.mkdir()

    if not (output_dir / 'for_paper').exists():
        (output_dir / 'for_paper').mkdir()

    output_dir = output_dir / 'for_paper'
    print('Starting pipeline')

    for im_id in tqdm(args.images):
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
            im_pil = Image.fromarray(predicted_mask)
            fixed_predicted_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))

            skeleton, segmented, segments, _ = skeletonize_v2(fixed_predicted_mask)
            pred_an_points, _ = find_anomality_points(fixed_predicted_mask)
            _image_with_all_anomalies = add_points_to_image(_image[y1:y2, x1:x2].copy(),
                                                            pred_an_points,
                                                            add_change=False,
                                                            color=(0, 255, 0))

            thickness = get_thickness_from_skeleton(fixed_predicted_mask, skeleton)

            def draw_circles_from_segments(img, thickness, segments, color, step=8, fill=False):
                new_image = img.copy()
                fill = -1 if fill else 1

                for seg in segments:
                    for p in seg[::step]:
                        p = p[0]
                        new_image = cv2.circle(new_image, (p[0], p[1]), round(thickness[p[1]][p[0]]), color, fill)

                return new_image

            img_with_circles = draw_circles_from_segments(image[y1:y2, x1:x2].copy(), thickness, segments, color=[0, 102, 0],
                                                          step=10)

            # img_with_circles = draw_circles_from_segments(image,
            #                                               step=10,
            #                                               thickness=thickness,
            #                                            segments=segments)

            def show_mask(mask, ax, random_color=False):
                if random_color:
                    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                else:
                    color = np.array([152 / 255, 251 / 255, 152 / 255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask_image)

            _img = img_with_circles.copy()
            _skel = skeleton
            _img[_skel != 0] = [255, 69, 0]
            print(take_largest_contour(mask[y1:y2, x1:x2]).shape)
            fontsize = 10
            plt.figure(figsize=(15, 8))
            # plt.subplots_adjust(wspace=0.01, hspace=0)

            plt.subplot(1, 4, 1)
            plt.imshow(image[y1:y2, x1:x2], cmap='gray')
            plt.yticks([])
            plt.xticks([])

            plt.subplot(1, 4, 2)
            plt.imshow(image[y1:y2, x1:x2], cmap='gray')
            show_mask(np.squeeze(take_largest_contour(mask[y1:y2, x1:x2])//255), plt.gca())
            # plt.imshow(image[y1:y2, x1:x2], cmap='gray')
            plt.yticks([])
            plt.xticks([])

            # plt.savefig(output_dir / f'{im_id}_{j}_real.png', transparent=True)
            # plt.close()
            # plt.title('(a)', fontsize=fontsize, y=-0.1)

            plt.subplot(1, 4, 3)
            plt.imshow(image[y1:y2, x1:x2], cmap='gray')
            show_mask(fixed_predicted_mask, plt.gca())
            # plt.imshow(predicted_mask, cmap='gray')
            plt.yticks([])
            plt.xticks([])
            # plt.title('(b)', fontsize=fontsize, y=-0.1)
            #
            # plt.subplot(1, 4, 3)
            # plt.imshow(np.logical_xor(skeleton, fixed_predicted_mask), cmap='gray')
            # plt.yticks([])
            # plt.xticks([])
            # plt.title('(c)', fontsize=fontsize, y=-0.1)

            plt.subplot(1, 4, 4)
            plt.imshow(_img)
            plt.yticks([])
            plt.xticks([])
            # plt.title('(d)', fontsize=fontsize, y=-0.1)
            # plt.show()
            plt.savefig(output_dir / f'{im_id}_{j}_anomalies.png', transparent=True)
            plt.close()


if __name__ == '__main__':
    main()
