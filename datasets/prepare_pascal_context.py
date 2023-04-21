# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import tqdm
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.io

def convert_pc59(mask_path, new_mask_path, pc59_dict):
    mat = scipy.io.loadmat(mask_path)
    mask = mat['LabelMap']

    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for trID, clsID in pc59_dict.items():
        mask_copy[mask == clsID] = trID

    min_value = np.amin(mask_copy)
    assert min_value >= 0, print(min_value)
    Image.fromarray(mask_copy).save(new_mask_path, "PNG")

def convert_pc459(mask_path, new_mask_path):
    mat = scipy.io.loadmat(mask_path)
    mask = mat['LabelMap']
    mask = mask - 1
    min_value = np.amin(mask)
    assert min_value >= 0, print(min_value)
    Image.fromarray(mask).save(new_mask_path, "TIFF")


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    print('Caution: we only generate the validation set!')
    pc_path = dataset_dir / "VOCdevkit/VOC2010"

    val_list = open(pc_path / "pascalcontext_val.txt", "r")
    pc459_labels = open(pc_path / "labels.txt", "r")
    pc59_labels = open(pc_path / "59_labels.txt", "r")

    pc459_dict = {}
    for line in pc459_labels.readlines():
        if ':' in line:
            idx, name = line.split(':')
            idx = int(idx.strip())
            name = name.strip()
            pc459_dict[name] = idx

    pc59_dict = {}
    for i, line in enumerate(pc59_labels.readlines()):
        name = line.split(':')[-1].strip()
        if name is not '':
            pc59_dict[i] = pc459_dict[name]

    pc459_dir = pc_path / "annotations_detectron2" / "pc459_val"
    pc459_dir.mkdir(parents=True, exist_ok=True)
    pc59_dir = pc_path / "annotations_detectron2" / "pc59_val"
    pc59_dir.mkdir(parents=True, exist_ok=True)

    for line in tqdm.tqdm(val_list.readlines()):
        fileid = line.strip()
        ori_mask = f'{pc_path}/trainval/{fileid}.mat'
        pc459_dst = f'{pc459_dir}/{fileid}.tif'
        pc59_dst = f'{pc59_dir}/{fileid}.png'
        if osp.exists(ori_mask):
            convert_pc459(ori_mask, pc459_dst)
            convert_pc59(ori_mask, pc59_dst, pc59_dict)
