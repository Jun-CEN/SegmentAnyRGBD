# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from collections import OrderedDict


# PATH to new clip model
clip_ckpt = torch.load('xx/open_clip/src/logs/2022_xx/checkpoints/epoch_x.pt')

new_model = OrderedDict()
state_dict = clip_ckpt['state_dict']

for k, v in state_dict.items():
    new_key = k.replace('module.','')
    new_model[new_key] = v

# PATH to trained ovseg model
ovseg_model = torch.load('xx/ovseg/output/model_final.pth', 'cpu')

for k, v in new_model.items():
    new_k = 'clip_adapter.clip_model.' + k
    if new_k in ovseg_model['model'].keys():
        ovseg_model['model'][new_k] = v
    else:
        print(f'{new_k} does not exist in ckpt')

# ovseg_model['model']['clip_adapter.clip_model.visual.mask_embedding'] = new_model['visual.mask_embedding']

torch.save(ovseg_model, 'xx/ovseg/output/ovseg_ft_mpt.pth')
