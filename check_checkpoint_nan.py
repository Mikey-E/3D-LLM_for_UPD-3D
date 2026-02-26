#!/usr/bin/env python3
import torch

print("Loading checkpoint...")
ckpt = torch.load('/project/3dllms/melgin/3D-LLM_for_UPD-3D/checkpoints/pretrain_blip2_sam_flant5xl_v2.pth', map_location='cpu')

model_state = ckpt.get('model', ckpt)

has_nan = False
has_inf = False

for k, v in model_state.items():
    if torch.is_tensor(v):
        if torch.isnan(v).any():
            print(f'NaN found in {k}')
            has_nan = True
        if torch.isinf(v).any():
            print(f'Inf found in {k}')
            has_inf = True

print(f'\nChecked {len(model_state)} keys')
print(f'Has NaN: {has_nan}')
print(f'Has Inf: {has_inf}')
