# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained swin model:
  wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl
  # Then, use swin_tiny_patch4_window7_224.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224.pkl"
INPUT:
  FORMAT: "RGB"
"""


def transform(path):
    model = torch.load(path, map_location="cpu")
    print(f"loading {path}......")
    state_dict = model["model"]
    state_dict = {
        k.replace("visual_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("visual_model")
    }
    source_keys = [k for k in state_dict.keys() if "relative_coords" in k]
    for k in source_keys:
        state_dict[
            k.replace("relative_coords", "relative_position_index")
        ] = state_dict[k]
        del state_dict[k]

    source_keys = [k for k in state_dict.keys() if "atten_mask_matrix" in k]
    for k in source_keys:
        state_dict[k.replace("atten_mask_matrix", "attn_mask")] = state_dict[k]
        del state_dict[k]

    source_keys = [k for k in state_dict.keys() if "rel_pos_embed_table" in k]
    for k in source_keys:
        state_dict[
            k.replace("rel_pos_embed_table", "relative_position_bias_table")
        ] = state_dict[k]
        del state_dict[k]

    source_keys = [k for k in state_dict.keys() if "channel_reduction" in k]
    for k in source_keys:
        state_dict[k.replace("channel_reduction", "reduction")] = state_dict[k]
        del state_dict[k]
    return {
        k if k.startswith("backbone.") else "backbone." + k: v
        for k, v in state_dict.items()
    }


if __name__ == "__main__":
    input = sys.argv[1]
    res = {
        "model": transform(input),
        "__author__": "third_party",
        "matching_heuristics": True,
    }
    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
