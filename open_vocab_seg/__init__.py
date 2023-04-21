# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from . import data
from . import modeling
from .config import add_ovseg_config

from .test_time_augmentation import SemanticSegmentorWithTTA
from .ovseg_model import OVSeg, OVSegDEMO
