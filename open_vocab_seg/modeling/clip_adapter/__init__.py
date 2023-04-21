# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from .text_template import (
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
)
from .adapter import ClipAdapter, MaskFormerClipAdapter


def build_text_prompt(cfg):
    if cfg.TEXT_TEMPLATES == "predefined":
        text_templates = PredefinedPromptExtractor(cfg.PREDEFINED_PROMPT_TEMPLATES)
    elif cfg.TEXT_TEMPLATES == "imagenet":
        text_templates = ImageNetPromptExtractor()
    elif cfg.TEXT_TEMPLATES == "vild":
        text_templates = VILDPromptExtractor()
    else:
        raise NotImplementedError(
            "Prompt learner {} is not supported".format(cfg.TEXT_TEMPLATES)
        )
    return text_templates
