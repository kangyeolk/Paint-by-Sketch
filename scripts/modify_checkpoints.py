import torch
import os
# pretrained_model_path='pretrained_models/sd-v1-4-full-ema-pruned.ckpt'
# pretrained_model_path='pretrained_models/tuned-global_step=2999.0-pruned.ckpt'
# pretrained_model_path='pretrained_models/wd-v1-3-full-opt.ckpt'
pretrained_model_path='checkpoints/model.ckpt'
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')
zero_data=torch.zeros(320,3,3,3)
new_weight=torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'],zero_data),dim=1)
ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight']=new_weight
out_name = pretrained_model_path.split('/')[-1][:-5] + '-modified-9channel.ckpt'
torch.save(ckpt_file, os.path.join("pretrained_models", out_name))