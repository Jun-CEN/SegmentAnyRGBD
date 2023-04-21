# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Feng Liang from https://github.com/MendelXu/zsseg.baseline/blob/master/datasets/prepare_voc_sem_seg.py

import os
import os.path as osp
from pathlib import Path
import tqdm

import numpy as np
from PIL import Image


clsID_to_trID = {
    0: 255,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    255: 255,
}

def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=clsID_to_trID, suffix=""
):
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = (
        osp.join(out_mask_dir, "train" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")



if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    print('Caution: we only generate the validation set!')
    voc_path = dataset_dir / "VOCdevkit" / "VOC2012"
    out_mask_dir = voc_path / "annotations_detectron2"
    out_image_dir = voc_path / "images_detectron2"
    for name in ["val"]:
        os.makedirs((out_mask_dir / name), exist_ok=True)
        os.makedirs((out_image_dir / name), exist_ok=True)
        val_list = [
            osp.join(voc_path, "SegmentationClassAug", f + ".png")
            for f in np.loadtxt(osp.join(voc_path, "ImageSets/Segmentation/val.txt"), dtype=np.str).tolist()
        ]
        for file in tqdm.tqdm(val_list):
            convert_to_trainID(file, out_mask_dir, is_train=False)
