# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import math
import time
import numpy as np

from Define import *
from Utils import *

# refer : https://github.com/makalo/CornerNet
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)

class CenterNet_Utils:
    def __init__(self, ):
        cx = np.arange(OUTPUT_WIDTH)
        cy = np.arange(OUTPUT_HEIGHT)
        cx, cy = np.meshgrid(cx, cy)

        self.centers = np.concatenate([cx[..., np.newaxis], cy[..., np.newaxis]], axis = -1)

    def Encode(self, image_shape, gt_bboxes, gt_classes):
        gt_heatmaps = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, CLASSES), dtype = np.float32)
        gt_offsets = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 2), dtype = np.float32)
        gt_sizes = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 2), dtype = np.float32)
        gt_masks = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 1), dtype = np.float32)

        h, w, c = image_shape
        w_ratio = OUTPUT_WIDTH / w
        h_ratio = OUTPUT_HEIGHT / h

        gt_bboxes = gt_bboxes * [w_ratio, h_ratio, w_ratio, h_ratio]
        for i in range(len(gt_bboxes)):
            xmin, ymin, xmax, ymax = gt_bboxes[i].astype(np.int32)
            gt_class = gt_classes[i]

            cx_norm = (xmin + xmax) / 2
            cy_norm = (ymin + ymax) / 2

            cx = int(cx_norm)
            cy = int(cy_norm)

            width = xmax - xmin
            height = ymax - ymin

            radius = gaussian_radius((height, width), 0.7)
            radius = max(0, int(radius))

            draw_gaussian(gt_heatmaps[..., gt_class], [cx, cy], radius)
            gt_offsets[cy, cx, :] = [cx_norm - cx, cy_norm - cy]
            gt_sizes[cy, cx, :] = [width / OUTPUT_WIDTH, height / OUTPUT_HEIGHT]
            gt_masks[cy, cx, 0] = 1.
        
        return gt_heatmaps, gt_offsets, gt_sizes, gt_masks

    def Decode(self, pred_heatmaps, pred_offsets, pred_sizes, detect_threshold = 0.05, top_k = 100):
        pred_sizes = pred_sizes * [OUTPUT_WIDTH, OUTPUT_HEIGHT]
        pred_centers = self.centers + pred_offsets
        
        pred_tl = pred_centers - pred_sizes / 2
        pred_rb = pred_centers + pred_sizes / 2
        pred_bboxes = np.concatenate([pred_tl, pred_rb], axis = -1)

        np.argsort(pred_heatmaps.reshape((-1)), axis = -1)

if __name__ == '__main__':
    centernet_utils = CenterNet_Utils()

    for data in np.load('./dataset/train_detection.npy', allow_pickle = True):
        image_name, gt_bboxes, gt_classes = data

        image_path = TRAIN_DIR + image_name
        image = cv2.imread(image_path)
        h, w, c = image.shape

        gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
        gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)

        gt_heatmaps = centernet_utils.Encode(image.shape, gt_bboxes, gt_classes)
        cv2.imshow(CLASS_NAMES[0], normalize_to_heatmap(gt_heatmaps[..., 0]))

        cv2.imshow('object', image)
        cv2.waitKey(0)
