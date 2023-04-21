# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Feng Liang from
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/modeling/clip_adapter/text_prompt.py
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/modeling/clip_adapter/utils.py

from typing import List

import clip
import torch
from torch import nn

IMAGENET_PROMPT = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


class ImageNetPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(IMAGENET_PROMPT)


class VILDPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(VILD_PROMPT)
