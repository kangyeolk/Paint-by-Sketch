# Following Deepfillv2 we binarize sketch inputs 

import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
import argparse

def sketch_magic_logic(sketch, thereshold=153): # follow deepfill v2 0.6 thereshold
    sketch[sketch > thereshold] = 255
    sketch[sketch <= thereshold] = 0
    return sketch

def open_image(path):
    obj = cv2.imread(path)
    obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch_root', type=str)
    parser.add_argument('--save_root', type=str)
    args = parser.parse_args()

    # e.g.,
    # sketch_root = '/home/nas2_userF/kangyeol/Project/webtoon2022/waifu-diffusion/danbooru_aesthetic/sketch/eval_results/imgs_epoch_019'
    # save_root = '/home/nas2_userF/kangyeol/Project/webtoon2022/waifu-diffusion/danbooru_aesthetic/sketch_bin'
    sketch_root = args.sketch_root
    save_root = args.save_root
    
    sketch_files = sorted(os.listdir(sketch_root))            
    os.makedirs(save_root, exist_ok=True)

    tgt_files = sketch_files
    print(f'Processing {len(tgt_files)} files...!')
    
    for idx, sketch_file in enumerate(tgt_files):
        sketch_file = os.path.join(sketch_root, sketch_file)
        save_name = os.path.join(save_root, os.path.basename(sketch_file))
        if os.path.isfile(save_name):            
            continue
        sketch = sketch_magic_logic(open_image(sketch_file))
        sketch = Image.fromarray(sketch)    
        sketch.save(save_name)
        if (idx + 1) % 100 == 0:
            print(f"{idx}/{len(tgt_files)}")