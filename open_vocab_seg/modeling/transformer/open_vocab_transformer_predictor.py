# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from torch import nn
from detectron2.config import configurable
from .transformer_predictor import TransformerPredictor, MLP


class OpenVocabTransformerPredictor(TransformerPredictor):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        embedding_dim: int,
        embed_hidden_dim: int,
        embed_layers: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        super().__init__(
            in_channels,
            False,
            num_classes=embedding_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            deep_supervision=deep_supervision,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
        )
        self.mask_classification = mask_classification
        # output FFNs
        if self.mask_classification:
            self.class_embed = MLP(
                hidden_dim, embed_hidden_dim, embedding_dim, embed_layers
            )

    def freeze_pretrained(self):
        for name, module in self.named_children():
            if name not in ["class_embed"]:
                for param in module.parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["embedding_dim"] = cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM
        ret["embed_hidden_dim"] = cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM
        ret["embed_layers"] = cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret
