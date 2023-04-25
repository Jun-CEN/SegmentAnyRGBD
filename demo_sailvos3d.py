# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo

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
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            start_time = time.time()
            predictions, visualized_output_rgb, visualized_output_depth, visualized_output_rgb_sam, visualized_output_depth_sam = demo.run_on_image_sam(path, class_names)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output_rgb.save('RGB_Semantic_SAM.png')
                visualized_output_depth.save('Depth_Semantic_SAM.png')
                visualized_output_rgb_sam.save('RGB_Semantic_SAM_Mask.png')
                visualized_output_depth_sam.save('Depth_Semantic_SAM_Mask.png')
                rgb_3d_sam = demo.get_xyzrgb('RGB_Semantic_SAM.png', path)
                depth_3d_sam = demo.get_xyzrgb('Depth_Semantic_SAM.png', path)
                rgb_3d_sam_mask = demo.get_xyzrgb('RGB_Semantic_SAM_Mask.png', path)
                depth_3d_sam_mask = demo.get_xyzrgb('Depth_Semantic_SAM_Mask.png', path)
                np.savez('xyzrgb.npz', rgb_3d_sam = rgb_3d_sam, depth_3d_sam = depth_3d_sam, rgb_3d_sam_mask = rgb_3d_sam_mask, depth_3d_sam_mask = depth_3d_sam_mask)
                demo.render_3d_video('xyzrgb.npz', path)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output_rgb.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        raise NotImplementedError
