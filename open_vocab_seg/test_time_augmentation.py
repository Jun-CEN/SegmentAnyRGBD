# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import copy
from itertools import count
import math
import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA
from detectron2.modeling.postprocessing import sem_seg_postprocess
import logging
from detectron2.utils.logger import log_every_n, log_first_n

__all__ = [
    "SemanticSegmentorWithTTA",
]


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _inference_with_model(self, inputs):
        if self.cfg.TEST.SLIDING_WINDOW:
            log_first_n(logging.INFO, "Using sliding window to test")

            outputs = []

            for input in inputs:
                image_size = input["image"].shape[1:]  # h,w
                if self.cfg.TEST.SLIDING_TILE_SIZE > 0:
                    tile_size = (
                        self.cfg.TEST.SLIDING_TILE_SIZE,
                        self.cfg.TEST.SLIDING_TILE_SIZE,
                    )
                else:
                    selected_mapping = {256: 224, 512: 256, 768: 512, 896: 512}
                    tile_size = min(image_size)
                    tile_size = selected_mapping[tile_size]
                    tile_size = (tile_size, tile_size)
                extra_info = {
                    k: v
                    for k, v in input.items()
                    if k not in ["image", "height", "width"]
                }
                log_every_n(
                    logging.INFO, "split {} to {}".format(image_size, tile_size)
                )
                overlap = self.cfg.TEST.SLIDING_OVERLAP
                stride = math.ceil(tile_size[0] * (1 - overlap))
                tile_rows = int(
                    math.ceil((image_size[0] - tile_size[0]) / stride) + 1
                )  # strided convolution formula
                tile_cols = int(math.ceil((image_size[1] - tile_size[1]) / stride) + 1)
                full_probs = None
                count_predictions = None
                tile_counter = 0

                for row in range(tile_rows):
                    for col in range(tile_cols):
                        x1 = int(col * stride)
                        y1 = int(row * stride)
                        x2 = min(x1 + tile_size[1], image_size[1])
                        y2 = min(y1 + tile_size[0], image_size[0])
                        x1 = max(
                            int(x2 - tile_size[1]), 0
                        )  # for portrait images the x1 underflows sometimes
                        y1 = max(
                            int(y2 - tile_size[0]), 0
                        )  # for very few rows y1 underflows

                        img = input["image"][:, y1:y2, x1:x2]
                        padded_img = nn.functional.pad(
                            img,
                            (
                                0,
                                tile_size[1] - img.shape[-1],
                                0,
                                tile_size[0] - img.shape[-2],
                            ),
                        )
                        tile_counter += 1
                        padded_input = {"image": padded_img}
                        padded_input.update(extra_info)
                        padded_prediction = self.model([padded_input])[0]["sem_seg"]
                        prediction = padded_prediction[
                            :, 0 : img.shape[1], 0 : img.shape[2]
                        ]
                        if full_probs is None:
                            full_probs = prediction.new_zeros(
                                prediction.shape[0], image_size[0], image_size[1]
                            )
                        if count_predictions is None:
                            count_predictions = prediction.new_zeros(
                                prediction.shape[0], image_size[0], image_size[1]
                            )
                        count_predictions[:, y1:y2, x1:x2] += 1
                        full_probs[
                            :, y1:y2, x1:x2
                        ] += prediction  # accumulate the predictions also in the overlapping regions

                full_probs /= count_predictions
                full_probs = sem_seg_postprocess(
                    full_probs,
                    image_size,
                    input.get("height", image_size[0]),
                    input.get("width", image_size[1]),
                )
                outputs.append({"sem_seg": full_probs})

            return outputs
        else:
            log_first_n(logging.INFO, "Using whole image to test")
            return self.model(inputs)

    def _batch_inference(self, batched_inputs):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.
        Inputs & outputs have the same format as :meth:`SemanticSegmentor.forward`
        """
        outputs = []
        inputs = []
        for idx, input in zip(count(), batched_inputs):
            inputs.append(input)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                with torch.no_grad():
                    outputs.extend(self._inference_with_model(inputs))
                inputs = []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                )  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # Delete now useless variables to avoid being out of memory
        del augmented_inputs
        # 2: merge the results
        # handle flip specially
        # outputs = [output.detach() for output in outputs]
        return self._merge_auged_output(outputs, tfms)

    def _merge_auged_output(self, outputs, tfms):
        new_outputs = []
        for output, tfm in zip(outputs, tfms):
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                new_outputs.append(output["sem_seg"].flip(dims=[2]))
            else:
                new_outputs.append(output["sem_seg"])
        del outputs
        # to avoid OOM with torch.stack
        final_predictions = new_outputs[0]
        for i in range(1, len(new_outputs)):
            final_predictions += new_outputs[i]
        final_predictions = final_predictions / len(new_outputs)
        del new_outputs
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms
