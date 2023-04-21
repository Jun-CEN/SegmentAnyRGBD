# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import multiprocessing as mp

import numpy as np
from PIL import Image

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

import gradio as gr

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def inference(class_names, input_img):
    mp.set_start_method("spawn", force=True)
    config_file = './configs/ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)

    demo = VisualizationDemo(cfg)

    class_names = class_names.split(',')
    img = read_image(input_img, format="BGR")
    _, visualized_output = demo.run_on_image(img, class_names)

    return Image.fromarray(np.uint8(visualized_output.get_image())).convert('RGB')

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()


examples = [['Oculus, Ukulele', './resources/demo_samples/sample_03.jpeg'],]
output_labels = ['segmentation map']

title = 'OVSeg'

description = """
Gradio Demo for Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP \n
You may click on of the examples or upload your own image. \n
OVSeg could perform open vocabulary segmentation, you may input more classes (seperate by comma).
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2210.04150' target='_blank'>
Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP
</a>
|
<a href='https://github.com' target='_blank'>Github Repo</a></p>
"""

gr.Interface(
    inference,
    inputs=[
        gr.inputs.Textbox(
            lines=1, placeholder=None, default='', label='class names'),
        gr.inputs.Image(type='filepath')
    ],
    outputs=gr.outputs.Image(label='segmentation map'),
    title=title,
    description=description,
    article=article,
    examples=examples).launch(enable_queue=True)
