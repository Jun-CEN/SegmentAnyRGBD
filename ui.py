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
import gradio as gr
from tools.util import *

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo, VisualizationDemoIndoor

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
        default=["/mnt/lustre/jkyang/PSG4D/sailvos3d/downloads/sailvos3d/trevor_1_int/images/000160.bmp"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        default=["person", "car", "motorcycle", "truck", "bird", "dog", "handbag", "suitcase", "bottle", "cup", "bowl", "chair", "potted plant", "bed", "dining table", "tv", "laptop", "cell phone", "bag", "bin", "box", "door", "road barrier", "stick", "lamp", "floor", "wall"],
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output", 
        default = "./pred",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "ovseg_swinbase_vitL14_ft_mpt.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser

args = get_parser().parse_args()

def greet_sailvos3d(rgb_input, depth_map_input, rage_matrices_input, class_candidates):
    print(args.class_names)
    print(class_candidates[0], class_candidates[1], class_candidates[2], class_candidates[3],)
    print(class_candidates.split(', '))
    args.input = [rgb_input]
    args.class_names = class_candidates.split(', ')
    depth_map_path = depth_map_input.name
    rage_matrices_path = rage_matrices_input.name
    print(args.input, args.class_names, depth_map_path, rage_matrices_path)
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    print(args.input)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            start_time = time.time()
            predictions, visualized_output_rgb, visualized_output_depth, visualized_output_rgb_sam, visualized_output_depth_sam = demo.run_on_image_sam(path, class_names, depth_map_path, rage_matrices_path)
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
                visualized_output_rgb.save('outputs/RGB_Semantic_SAM.png')
                visualized_output_depth.save('outputs/Depth_Semantic_SAM.png')
                visualized_output_rgb_sam.save('outputs/RGB_Semantic_SAM_Mask.png')
                visualized_output_depth_sam.save('outputs/Depth_Semantic_SAM_Mask.png')
                rgb_3d_sam = demo.get_xyzrgb('outputs/RGB_Semantic_SAM.png', depth_map_path, rage_matrices_path)
                depth_3d_sam = demo.get_xyzrgb('outputs/Depth_Semantic_SAM.png', depth_map_path, rage_matrices_path)
                rgb_3d_sam_mask = demo.get_xyzrgb('outputs/RGB_Semantic_SAM_Mask.png', depth_map_path, rage_matrices_path)
                depth_3d_sam_mask = demo.get_xyzrgb('outputs/Depth_Semantic_SAM_Mask.png', depth_map_path, rage_matrices_path)
                np.savez('outputs/xyzrgb.npz', rgb_3d_sam = rgb_3d_sam, depth_3d_sam = depth_3d_sam, rgb_3d_sam_mask = rgb_3d_sam_mask, depth_3d_sam_mask = depth_3d_sam_mask)
                demo.render_3d_video('outputs/xyzrgb.npz', depth_map_path)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output_rgb.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        raise NotImplementedError
    
    Depth_Semantic_SAM_Mask = read_image('outputs/Depth_Semantic_SAM_Mask.png')
    RGB_Semantic_SAM_Mask = read_image('outputs/RGB_Semantic_SAM_Mask.png')
    Depth_map = read_image('outputs/Depth_rendered.png')
    Depth_Semantic_SAM_Mask_gif = 'outputs/depth_3d_sam_mask.gif'
    RGB_Semantic_SAM_Mask_gif = 'outputs/rgb_3d_sam_mask.gif'
    return RGB_Semantic_SAM_Mask, RGB_Semantic_SAM_Mask_gif, Depth_map, Depth_Semantic_SAM_Mask, Depth_Semantic_SAM_Mask_gif

def greet_scannet(rgb_input, depth_map_input, class_candidates):
    rgb_input = rgb_input
    depth_map_input = depth_map_input.name
    class_candidates = class_candidates.split(', ')
    print(rgb_input, depth_map_input, class_candidates)
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemoIndoor(cfg)
    from datasets.scannet_preprocess.meta_data.scannet200_constants import CLASS_LABELS_200
    class_names = list(CLASS_LABELS_200)
    """ args.input = glob.glob(os.path.expanduser(args.input[0]))
    assert args.input, "The input path(s) was not found" """
    start_time = time.time()
    predictions, output2D, output3D = demo.run_on_pcd_ui(rgb_input, depth_map_input, class_candidates)

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

    Depth_Semantic_SAM_Mask = read_image('outputs/Depth_Semantic_SAM_Mask.png')
    RGB_Semantic_SAM_Mask = read_image('outputs/RGB_Semantic_SAM_Mask.png')
    Depth_map = read_image('outputs/Depth_rendered.png')
    Depth_Semantic_SAM_Mask_gif = 'outputs/depth_3d_sam_mask.gif'
    RGB_Semantic_SAM_Mask_gif = 'outputs/rgb_3d_sam_mask.gif'
    return RGB_Semantic_SAM_Mask, RGB_Semantic_SAM_Mask_gif, Depth_map, Depth_Semantic_SAM_Mask, Depth_Semantic_SAM_Mask_gif


with gr.Blocks(analytics_enabled=False) as segrgbd_iface:
        gr.Markdown("<div align='center'> <h2> Semantic Segment AnyRGBD </span> </h2> \
                     <a style='font-size:18px;color: #000000' href='https://github.com/Jun-CEN/SegmentAnyRGBD'> Github </div>")
        
        #######t2v#######
        with gr.Tab(label="Dataset: Sailvos3D"):
            with gr.Column():
                with gr.Row():
                    # with gr.Tab(label='input'):
                    with gr.Column():
                        with gr.Row():
                            Input_RGB_Component = gr.Image(label = 'RGB_Input', type = 'filepath').style(width=320, height=200)
                            Depth_Map_Output_Component = gr.Image(label = "Depth_Map").style(width=320, height=200)
                        with gr.Row():
                            Depth_Map_Input_Component = gr.File(label = 'Depth_map')
                            Component_2D_to_3D_Projection_Parameters = gr.File(label = '2D_to_3D_Projection_Parameters')
                        with gr.Row():
                            Class_Candidates_Component = gr.Text(label = 'Class_Candidates')
                        vc_end_btn = gr.Button("Send")
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            RGB_Semantic_SAM_Mask_Component = gr.Image(label = "RGB_Semantic_SAM_Mask").style(width=320, height=200)
                            RGB_Semantic_SAM_Mask_3D_Component = gr.Image(label = "3D_RGB_Semantic_SAM_Mask").style(width=320, height=200)
                        with gr.Row():
                            Depth_Semantic_SAM_Mask_Component = gr.Image(label = "Depth_Semantic_SAM_Mask").style(width=320, height=200)
                            Depth_Semantic_SAM_Mask_3D_Component = gr.Image(label = "3D_Depth_Semantic_SAM_Mask").style(width=320, height=200)
                gr.Examples(examples=[
                        [
                            'UI/sailvos3d/ex1/inputs/rgb_000160.bmp',
                            'UI/sailvos3d/ex1/inputs/depth_000160.npy',
                            'UI/sailvos3d/ex1/inputs/rage_matrices_000160.npz',
                            'person, car, motorcycle, truck, bird, dog, handbag, suitcase, bottle, cup, bowl, chair, potted plant, bed, dining table, tv, laptop, cell phone, bag, bin, box, door, road barrier, stick, lamp, floor, wall',
                        ],
                        [
                            'UI/sailvos3d/ex2/inputs/rgb_000540.bmp',
                            'UI/sailvos3d/ex2/inputs/depth_000540.npy',
                            'UI/sailvos3d/ex2/inputs/rage_matrices_000540.npz',
                            'person, car, motorcycle, truck, bird, dog, handbag, suitcase, bottle, cup, bowl, chair, potted plant, bed, dining table, tv, laptop, cell phone, bag, bin, box, door, road barrier, stick, lamp, floor, wall',
                        ]],
                            inputs=[Input_RGB_Component, Depth_Map_Input_Component, Component_2D_to_3D_Projection_Parameters, Class_Candidates_Component],
                            outputs=[RGB_Semantic_SAM_Mask_Component, RGB_Semantic_SAM_Mask_3D_Component, Depth_Map_Output_Component, Depth_Semantic_SAM_Mask_Component, Depth_Semantic_SAM_Mask_3D_Component],
                            fn=greet_sailvos3d)
            vc_end_btn.click(inputs=[Input_RGB_Component, Depth_Map_Input_Component, Component_2D_to_3D_Projection_Parameters, Class_Candidates_Component],
                            outputs=[RGB_Semantic_SAM_Mask_Component, RGB_Semantic_SAM_Mask_3D_Component, Depth_Map_Output_Component, Depth_Semantic_SAM_Mask_Component, Depth_Semantic_SAM_Mask_3D_Component],
                            fn=greet_sailvos3d)
        
        with gr.Tab(label="Dataset: Scannet"):
            with gr.Column():
                with gr.Row():
                    # with gr.Tab(label='input'):
                    with gr.Column():
                        with gr.Row():
                            Input_RGB_Component = gr.Image(label = 'RGB_Input', type = 'filepath').style(width=320, height=200)
                            Depth_Map_Output_Component = gr.Image(label = "Depth_Map").style(width=320, height=200)
                        with gr.Row():
                            Depth_Map_Input_Component = gr.File(label = "Depth_Map")
                            Class_Candidates_Component = gr.Text(label = 'Class_Candidates')
                        vc_end_btn = gr.Button("Send")
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            RGB_Semantic_SAM_Mask_Component = gr.Image(label = "RGB_Semantic_SAM_Mask").style(width=320, height=200)
                            RGB_Semantic_SAM_Mask_3D_Component = gr.Image(label = "3D_RGB_Semantic_SAM_Mask").style(width=320, height=200)
                        with gr.Row():
                            Depth_Semantic_SAM_Mask_Component = gr.Image(label = "Depth_Semantic_SAM_Mask").style(width=320, height=200)
                            Depth_Semantic_SAM_Mask_3D_Component = gr.Image(label = "3D_Depth_Semantic_SAM_Mask").style(width=320, height=200)
                gr.Examples(examples=[
                        [
                            'UI/scannetv2/examples/scene0000_00/color/1660.jpg',
                            'UI/scannetv2/examples/scene0000_00/depth/1660.png',
                            'wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub, other furniture',
                        ],
                        [
                            'UI/scannetv2/examples/scene0000_00/color/5560.jpg',
                            'UI/scannetv2/examples/scene0000_00/depth/5560.png',
                            'wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub, other furniture',
                        ]],
                            inputs=[Input_RGB_Component, Depth_Map_Input_Component, Class_Candidates_Component],
                            outputs=[RGB_Semantic_SAM_Mask_Component, RGB_Semantic_SAM_Mask_3D_Component, Depth_Map_Output_Component, Depth_Semantic_SAM_Mask_Component, Depth_Semantic_SAM_Mask_3D_Component],
                            fn=greet_scannet)
            vc_end_btn.click(inputs=[Input_RGB_Component, Depth_Map_Input_Component, Class_Candidates_Component],
                            outputs=[RGB_Semantic_SAM_Mask_Component, RGB_Semantic_SAM_Mask_3D_Component, Depth_Map_Output_Component, Depth_Semantic_SAM_Mask_Component, Depth_Semantic_SAM_Mask_3D_Component],
                            fn=greet_scannet)


demo = segrgbd_iface
demo.launch()

