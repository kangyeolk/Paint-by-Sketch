from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class AestheticSketchCaptionDataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)

        if state == 'train':
            self.random_trans=A.Compose([
                A.Resize(height=224,width=224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20),
                A.Blur(p=0.3),
                A.ElasticTransform(p=0.3)
            ])
        else:
            self.random_trans=A.Compose([
                    A.Resize(height=224,width=224)])

        self.resize_only=A.Compose([
            A.Resize(height=224,width=224)
        ]) 
        
        self.images_path_list=[]
        images_dir=os.path.join(args['dataset_dir'], 'images') 
        per_dir_file_list=os.listdir(images_dir)        
        
        random.seed(123123)
        random.shuffle(per_dir_file_list)
        
        break_point = int(0.9*len(per_dir_file_list))
        train_list = per_dir_file_list[:break_point]
        validation_list = per_dir_file_list[break_point:]

        if state == "train":
            for file_name in train_list:
                self.images_path_list.append(os.path.join(images_dir, file_name))
        elif state == "validation":
            for file_name in validation_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))
        else:
            for file_name in validation_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))
        
        self.images_path_list.sort()
        self.length=len(self.images_path_list)
     
    # generate random masks
    def random_bbox(self, im_shape, ratio=1, mask_full_image=False):
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
        # use this to always mask the whole image
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
        bbox = [center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2]
        return bbox

    def __getitem__(self, index):
        
        if self.state != 'train':
            random.seed(index*77) # fix seed for consistent mask
        
        img_path=self.images_path_list[index]
        file_name=os.path.splitext(os.path.basename(img_path))[0]+'.jpg'
        sketch_path = os.path.join('/'.join(img_path.split('/')[:-1]).replace('images', 'sketch_bin'), file_name.replace('jpg', 'png'))
        
        img_p = Image.open(img_path).convert("RGB")
        sketch_p = Image.open(sketch_path).convert("RGB")
        
        if self.args["ref_type"] == 'cmap':
            cmap_path = os.path.join(os.path.join('/'.join(img_path.split('/')[:-1])).replace('images', 'colormap_k8'), file_name)
        elif self.args["ref_type"] == 'cmap_w_sketch':
            cmap_path = os.path.join(os.path.join('/'.join(img_path.split('/')[:-1])).replace('images', 'colormap_with_sketch'), file_name.replace('jpg', 'png'))
            
        ### Get reference image
        # prob=random.uniform(0, 1)
        # if prob < 0.5:
        bbox=self.random_bbox(img_p.size,1,False)         
        # else:
        #     bbox=self.random_bbox(img_p.size,1,True)
        
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np=cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)

        if self.args["ref_type"] == 'rgb':
            ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
            ref_image_tensor=self.random_trans(image=ref_image_tensor)
            ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor=get_tensor_clip()(ref_image_tensor)
        elif self.args["ref_type"] in ['cmap', 'cmap_w_sketch']:
            cmap_p_np = cv2.imread(cmap_path)
            cmap_p_np = cv2.cvtColor(cmap_p_np, cv2.COLOR_BGR2RGB)
            cmap_p_np_cropped=cmap_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
            ref_image_tensor = self.resize_only(image=cmap_p_np_cropped)
            ref_image_pil = Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor = get_tensor_clip()(ref_image_pil)

        txt_path = os.path.join('/'.join(img_path.split('/')[:-1]).replace('images', 'txt'), file_name.replace('jpg', 'txt'))
        f = open(txt_path, 'r')
        read_txt = f.readline()    
        prompts = read_txt.replace(' ', ',')
        f.close()        
        txt = prompts

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        sketch_tensor = get_tensor()(sketch_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]

        # debug- at least 0
        left_freespace = max(left_freespace, 0)
        right_freespace = max(right_freespace, 0)
        up_freespace = max(up_freespace, 0)
        down_freespace = max(down_freespace, 0)

        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
                sketch_tensor_cropped=sketch_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                sketch_tensor_cropped=sketch_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
                sketch_tensor_cropped=sketch_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                sketch_tensor_cropped=sketch_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor
            sketch_tensor_cropped=sketch_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        sketch_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(sketch_tensor_cropped)
        
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        # predict full sketch
        data = {"GT":image_tensor_resize,
                "inpaint_image":inpaint_tensor_resize,
                "inpaint_mask":mask_tensor_resize,
                "ref_imgs":ref_image_tensor,
                "sketch": sketch_tensor_resize,
                'txt': txt}
        return data

    def __len__(self):
        return self.length



class AestheticNoSketchCaptionDataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)

        
        self.random_trans=A.Compose([
                A.Resize(height=224,width=224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20),
                A.Blur(p=0.3),
                A.ElasticTransform(p=0.3)
            ])        

        self.resize_only=A.Compose([
            A.Resize(height=224,width=224)
        ]) 

        self.images_path_list=[]
        images_dir=os.path.join(args['dataset_dir'], 'images') 
        per_dir_file_list=os.listdir(images_dir)        
        
        random.seed(123123)
        random.shuffle(per_dir_file_list)
        
        break_point = int(0.9*len(per_dir_file_list))
        train_list = per_dir_file_list[:break_point]
        validation_list = per_dir_file_list[break_point:]

        if state == "train":
            for file_name in train_list:
                self.images_path_list.append(os.path.join(images_dir, file_name))
        elif state == "validation":
            for file_name in validation_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))
        else:
            for file_name in validation_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))
        
        self.images_path_list.sort()
        self.length=len(self.images_path_list)
     
    # generate random masks
    def random_bbox(self, im_shape, ratio=1, mask_full_image=False):
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
        # use this to always mask the whole image
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))

        bbox = [center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2]
        return bbox

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)


    def __getitem__(self, index):
        
        if self.state != 'train':
            random.seed(index*77) # fix seed for consistent mask
        
        img_path=self.images_path_list[index]
        file_name=os.path.splitext(os.path.basename(img_path))[0]+'.jpg'
                
        img_p = Image.open(img_path).convert("RGB")
        
        
        if self.args["ref_type"] == 'cmap':
            cmap_path = os.path.join(os.path.join('/'.join(img_path.split('/')[:-1])).replace('images', 'colormap_k8'), file_name)
        elif self.args["ref_type"] == 'cmap_w_sketch':
            cmap_path = os.path.join(os.path.join('/'.join(img_path.split('/')[:-1])).replace('images', 'colormap_with_sketch'), file_name.replace('jpg', 'png'))
            
        ### Get reference image
        bbox=self.random_bbox(img_p.size,1,False)         
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np=cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)

        if self.args["ref_type"] == 'rgb':
            ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
            ref_image_tensor=self.random_trans(image=ref_image_tensor)
            ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor=get_tensor_clip()(ref_image_tensor)
        elif self.args["ref_type"] in ['cmap', 'cmap_w_sketch']:
            cmap_p_np = cv2.imread(cmap_path)
            cmap_p_np = cv2.cvtColor(cmap_p_np, cv2.COLOR_BGR2RGB)
            cmap_p_np_cropped=cmap_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
            ref_image_tensor = self.resize_only(image=cmap_p_np_cropped)
            ref_image_pil = Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor = get_tensor_clip()(ref_image_pil)

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]

        # debug- at least 0
        left_freespace = max(left_freespace, 0)
        right_freespace = max(right_freespace, 0)
        up_freespace = max(up_freespace, 0)
        down_freespace = max(down_freespace, 0)

        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]                
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor                
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)        
        
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        # predict full sketch
        data = {"GT":image_tensor_resize,
                "inpaint_image":inpaint_tensor_resize,
                "inpaint_mask":mask_tensor_resize,
                "ref_imgs":ref_image_tensor}
        return data

    def __len__(self):
        return self.length



if __name__ == '__main__':
    pass


