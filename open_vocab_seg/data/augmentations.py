# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import numbers
import numpy as np
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import (
    CropTransform,
    ResizeTransform,
    TransformList,
)
from PIL import Image
from fvcore.transforms.transform import PadTransform


def mask2box(mask: np.ndarray):
    # use naive way
    row = np.nonzero(mask.sum(axis=0))[0]
    if len(row) == 0:
        return None
    x1 = row.min()
    x2 = row.max()
    col = np.nonzero(mask.sum(axis=1))[0]
    y1 = col.min()
    y2 = col.max()
    return x1, y1, x2 + 1 - x1, y2 + 1 - y1


def expand_box(x, y, w, h, expand_ratio=1.0, max_h=None, max_w=None):
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1] = max(0, box[1])
        box[3] = min(max_h - 1, box[3])
    if max_w is not None:
        box[0] = max(0, box[0])
        box[2] = min(max_w - 1, box[2])
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    return [int(b) for b in box]


class CropImageWithMask(Augmentation):
    def __init__(self, expand_ratio=1.0, mode="choice"):
        if isinstance(expand_ratio, numbers.Number):
            expand_ratio = (expand_ratio, expand_ratio)
        self.mode = mode
        self.expand_ratio = expand_ratio
        if self.mode == "range":
            assert len(expand_ratio) == 2 and expand_ratio[0] < expand_ratio[1]

    def get_transform(self, image, sem_seg, category_id):
        input_size = image.shape[:2]
        bin_mask = sem_seg == category_id
        x, y, w, h = mask2box(bin_mask)
        if self.mode == "choice":
            expand_ratio = np.random.choice(self.expand_ratio)
        else:
            expand_ratio = np.random.uniform(self.expand_ratio[0], self.expand_ratio[1])
        x, y, w, h = expand_box(x, y, w, h, expand_ratio, *input_size)
        w = max(w, 1)
        h = max(h, 1)
        return CropTransform(x, y, w, h, input_size[1], input_size[0])


class CropImageWithBox(Augmentation):
    def __init__(self, expand_ratio=1.0, mode="choice"):
        if isinstance(expand_ratio, numbers.Number):
            expand_ratio = (expand_ratio, expand_ratio)
        self.mode = mode
        self.expand_ratio = expand_ratio
        if self.mode == "range":
            assert len(expand_ratio) == 2 and expand_ratio[0] < expand_ratio[1]

    def get_transform(self, image, boxes):
        input_size = image.shape[:2]
        x, y, x2, y2 = boxes[0]
        w = x2 - x + 1
        h = y2 - y + 1
        if self.mode == "choice":
            expand_ratio = np.random.choice(self.expand_ratio)
        else:
            expand_ratio = np.random.uniform(self.expand_ratio[0], self.expand_ratio[1])
        x, y, w, h = expand_box(x, y, w, h, expand_ratio, *input_size)
        w = max(w, 1)
        h = max(h, 1)
        return CropTransform(x, y, w, h, input_size[1], input_size[0])


class RandomResizedCrop(Augmentation):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=Image.BILINEAR,
    ):
        if isinstance(size, int):
            size = (size, size)
        else:
            assert isinstance(size, (tuple, list)) and len(size) == 2

        self.size = size

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_transform(self, image):
        height, width = image.shape[:2]
        area = height * width

        log_ratio = np.log(np.array(self.ratio))
        is_success = False
        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, width - w + 1)
                j = np.random.randint(0, height - h + 1)

                is_success = True
                break

        if not is_success:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (width - w) // 2
            j = (height - h) // 2
        return TransformList(
            [
                CropTransform(i, j, w, h, width, height),
                ResizeTransform(
                    h, w, self.size[1], self.size[0], interp=self.interpolation
                ),
            ]
        )


class CenterCrop(Augmentation):
    def __init__(self, size, seg_ignore_label):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            size = (size[0], size[0])
        self.size = size
        self.seg_ignore_label = seg_ignore_label

    def get_transform(self, image):

        image_height, image_width = image.shape[:2]
        crop_height, crop_width = self.size

        transforms = []
        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2
                if crop_height > image_height
                else 0,
            ]
            transforms.append(
                PadTransform(
                    *padding_ltrb,
                    orig_w=image_width,
                    orig_h=image_height,
                    seg_pad_value=self.seg_ignore_label
                )
            )
            image_width, image_height = (
                image_width + padding_ltrb[0] + padding_ltrb[2],
                image_height + padding_ltrb[1] + padding_ltrb[3],
            )

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        transforms.append(
            CropTransform(
                crop_left, crop_top, crop_width, crop_height, image_width, image_height
            )
        )
        return TransformList(transforms)
