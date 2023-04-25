# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import matplotlib as mpl
from tools.util import *

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemoIndoor
# from sam3d import *
        
# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    """ parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    ) """
    """ parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    ) """
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--rgb_path', type=str, default='UI/scannetv2/examples', help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='UI/scannetv2', help='the path of pointcload data')
    parser.add_argument('--color_name', type=str, default='5560.jpg', help='the name of segment image')
    parser.add_argument('--merge_pcd', action ='store_true', help='merge point cloud or not')
    parser.add_argument('--save_path', type=str, default='', help='Where to save the pcd results')
    parser.add_argument('--scannetv2_train_path', type=str, default='datasets/scannet_preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='datasets/scannet_preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemoIndoor(cfg)
    from datasets.scannet_preprocess.meta_data.scannet200_constants import CLASS_LABELS_200
    class_names = list(CLASS_LABELS_200)
    if True:
        if True:
            """ args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found" """
            voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "sem_map"))
            # if not os.path.exists(args.save_path):
            #     os.makedirs(args.save_path)
            scene_names = sorted(os.listdir(args.rgb_path))
            for scene_name in scene_names:
                # use PIL, to be consistent with evaluation
                if os.path.exists(os.path.join(args.save_path, scene_name + ".pth")):
                    continue
                if args.color_name == '':
                    color_names = sorted(os.listdir(os.path.join(args.rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]), reverse=True)
                else: 
                    color_names = [args.color_name]
                pcd_list = []
                for i, color_name in enumerate(color_names):
                    path = os.path.join(args.rgb_path, scene_name, 'color', color_name)
                    start_time = time.time()
                    print(color_name, flush=True)
                    predictions, output2D, output3D = demo.run_on_pcd(args.rgb_path, scene_name, color_name, class_names)
                    pcd_dict = output3D['pcd_depth']
                    if len(pcd_dict["coord"]) == 0:
                        continue
                    pcd_dict = voxelize(pcd_dict)
                    pcd_list.append(pcd_dict)
                    
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )

                    if True:
                        output2D['sem_seg_on_rgb'].save('outputs/RGB_Semantic_SAM.png')
                        output2D['sem_seg_on_depth'].save('outputs/Depth_Semantic_SAM.png')
                        output2D['sam_seg_on_rgb'].save('outputs/RGB_Semantic_SAM_Mask.png')
                        output2D['sam_seg_on_depth'].save('outputs/Depth_Semantic_SAM_Mask.png')
                        """ rgb_3d_sam = demo.get_xyzrgb('outputs/RGB_Semantic_SAM.png', path)
                        depth_3d_sam = demo.get_xyzrgb('outputs/Depth_Semantic_SAM.png', path)
                        rgb_3d_sam_mask = demo.get_xyzrgb('outputs/RGB_Semantic_SAM_Mask.png', path)
                        depth_3d_sam_mask = demo.get_xyzrgb(outputs/'Depth_Semantic_SAM_Mask.png', path) """
                        rgb_3d_sem = output3D['rgb_3d_sem']
                        depth_3d_sem = output3D['depth_3d_sem']
                        rgb_3d_sam = output3D['rgb_3d_sam']
                        depth_3d_sam = output3D['depth_3d_sam']
                        
                        np.savez('outputs/xyzrgb.npz', rgb_3d_sam = rgb_3d_sem, depth_3d_sam = depth_3d_sem, rgb_3d_sam_mask = rgb_3d_sam, depth_3d_sam_mask = depth_3d_sam)
                        demo.render_3d_video('outputs/xyzrgb.npz')
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, output2D['sem_seg_on_rgb'].get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
                if args.merge_pcd:
                    with open(args.scannetv2_train_path) as train_file:
                        train_scenes = train_file.read().splitlines()
                    with open(args.scannetv2_val_path) as val_file:
                        val_scenes = val_file.read().splitlines()

                    if scene_name in train_scenes:
                        scene_path = os.path.join(args.data_path, "train", scene_name + ".pth")
                    elif scene_name in val_scenes:
                        scene_path = os.path.join(args.data_path, "val", scene_name + ".pth")

                    demo.merge_pcd(pcd_list, args.data_path, args.save_path, scene_path, args.voxel_size, args.th)
    else:
        raise NotImplementedError