import pandas as pd
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas

import argparse, os, sys, glob
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import cv2

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import torchvision

# sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))
sys.path.insert(0, os.getcwd())
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from torchvision.transforms import Resize
from easydict import EasyDict as edict
from matplotlib import colors
from datetime import datetime

##### Utils #####

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model, pl_sd['global_step']


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

def un_norm(x):
    return (x+1.0)/2.0

def get_input_sketch(config, sketch_tensor, mask):
    sketch_stage_config = config.model['params']['sketch_stage_config']
    if sketch_stage_config['input_edge_type'] == 'full':           
        in_sketch = sketch_tensor
    elif sketch_stage_config['input_edge_type'] == 'only_mask_region':           
        in_sketch = ((sketch_tensor+1)/2) * (1 - mask)
        in_sketch = 2*in_sketch-1
    elif sketch_stage_config['input_edge_type'] == 'erase_mask_region':           
        in_sketch = ((sketch_tensor+1)/2) * mask
        in_sketch = 2*in_sketch-1    
    return in_sketch

def make_grid(cols,rows):
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid

def coloring(x, color_name):    
    rgb = colors.hex2color(colors.cnames[f'{color_name}'])
    x = x.astype(np.float64)
    x[:, :, 0] *= rgb[0]
    x[:, :, 1] *= rgb[1]
    x[:, :, 2] *= rgb[2]
    x = x.astype(np.uint8)    
    return x

def preprocess_ref_canvas_bg(mask_p, sketch_p):    
    mask_p.convert('RGB').save('demo/ref_mask_pl.png')
    sketch_p.convert('RGB').save('demo/ref_sketch_p.png')
    tmp_mask_np = np.array(mask_p.convert('L'))    
    tmp_sketch_np = np.array(sketch_p.convert('RGB'))
    # print(tmp_mask_np.shape)
    # print(tmp_sketch_np.shape)
    # print((tmp_mask_np == 255).sum())
    # print((tmp_mask_np == 0).sum())

    # Reversely compute bounding box    
    xs, ys = np.where(tmp_mask_np == 255)
    max_x = min(xs.max()+10, 512)
    min_x = max(0, xs.min()-10)
    max_y = min(ys.max()+10, 512)
    min_y = max(0, ys.min()-10)
    bbox = [min_x, min_y, max_x, max_y]
    

    tmp_sketch_np = 255 - tmp_sketch_np # black edge, white background 
    tmp_sketch_np_cropped=tmp_sketch_np[bbox[0]:bbox[2],bbox[1]:bbox[3],:]
    # tmp_sketch_np_cropped=tmp_sketch_np[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    # tmp_sketch_np_cropped[tmp_sketch_np_cropped > 127] = 255
    # tmp_sketch_np_cropped[tmp_sketch_np_cropped < 127] = 0    
    ref_p_path = 'demo/third_canvas_data.png'
    Image.fromarray(tmp_sketch_np_cropped).resize((512,512)).save(ref_p_path)
    return ref_p_path


##### Inference #####
def paint_with_sketch(
        img_p, mask, sketch_p, ref_p, text_input, control_config, one_time_load):
    
    # Model load
    model = one_time_load['model']
    config = one_time_load['config']
    sampler = one_time_load['sampler']
    opt = one_time_load['opt']
    device = one_time_load['device']
    
    img_p = img_p.convert("RGB").resize((512, 512))
    mask = mask.convert("L").resize((512, 512))
    sketch_p = sketch_p.convert("RGB").resize((512, 512))

    if opt.ref_type in ['cmap', 'camp_with_edge']:
        ref_p = np.array(ref_p.convert("RGB"))
        if opt.ref_type == 'camp_with_edge':                        
            drawn_sketch = cv2.imread('demo/third_canvas_data.png')
            drawn_sketch = cv2.cvtColor(drawn_sketch, cv2.COLOR_BGR2RGB)    
            ref_p[ref_p == 0] = 255 # flip to white background while preserving colormap
            ref_p[drawn_sketch == 0] = 0 # add sketch part as black
        ref_p = Image.fromarray(ref_p)
    else:
        ref_p = Image.open('demo/ref_p.png').convert('RGB')
    
    # For debug
    Image.blend(img_p, mask.convert('RGB'), 0.5).save('demo/overlayed.png')
    img_p.save('demo/img_p.png')
    mask.convert('RGB').save('demo/mask.jpg')
    sketch_p.convert('RGB').save('demo/sketch_p.png')    
    ref_p.save('demo/ref_p.png')
    
    image_tensor = get_tensor()(img_p)
    image_tensor = image_tensor.unsqueeze(0)
    ref_p = ref_p.resize((224,224))
    ref_tensor=get_tensor_clip()(ref_p)
    ref_tensor = ref_tensor.unsqueeze(0)
    mask = np.array(mask)[None,None]
    mask = 1 - mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask)
    inpaint_image = image_tensor*mask_tensor    
    sketch_tensor = get_tensor()(sketch_p)
    sketch_tensor = sketch_tensor.unsqueeze(0)
    sketch_tensor = get_input_sketch(config, sketch_tensor, mask)    
    
    # To gpu
    test_model_kwargs={}
    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
    test_model_kwargs['in_sketch']=sketch_tensor.to(device) 
    ref_tensor = ref_tensor.to(device)
    # prompts = [text_input]

    # Control vars
    scale = control_config['scale']
    not_use_sketch_after = control_config['not_use_sketch_after']
    test_model_kwargs['sketch_config'] = {}
    test_model_kwargs['sketch_config']['not_use_sketch_after'] = not_use_sketch_after

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                if opt.ref_type in ['rgb', 'cmap', 'camp_with_edge']:                
                    if scale != 1.0:
                        uc = model.learnable_vector                    
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = model.proj_out(c)
                elif opt.ref_type == 'txt':                    
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(1 * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)                    
                    c = model.get_learned_conditioning(prompts)

                z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                test_model_kwargs['inpaint_image']=z_inpaint
                test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])
                test_model_kwargs['in_sketch']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['in_sketch'])

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=None,
                                                    test_model_kwargs=test_model_kwargs)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = 255. * x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()                

    ret_img = x_samples_ddim.astype(np.uint8)[0]
    ret_img = Image.fromarray(ret_img)
    ret_img.save('demo/ret_img.png')

    return ret_img

@st.experimental_singleton
def get_model():
    opt = edict({
        # Variables
        'gpu':  0,
        'ckpt': 'Cartoon_v1_aesthetic_image',                
        'ref_type': 'rgb',
        'ddim_steps': 50,
        'ddim_eta': 0.0,
        'plms': True,
        'H': 512,
        'W': 512,
        'seed': 321,

        # Fixed
        'n_samples': 1,
        'C': 4,
        'f': 8,
        'scale': 5.0,   # better than 0.5 
        'precision': 'autocast',        
    })
    
    seed_everything(opt.seed)
    
    log_ret = os.path.join('models', opt.ckpt)
    run_name = sorted(os.listdir(log_ret))[-1]
    config_name = [c for c in os.listdir(os.path.join(log_ret, run_name, 'configs')) if 'project' in c][0]
    
    # Overwrite ckpt, config
    opt.ckpt = os.path.join(log_ret, run_name, 'checkpoints', 'last.ckpt')
    opt.config = os.path.join(log_ret, run_name, 'configs', config_name)
    
    # GPU    
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

    sampler, model, config = None, None, None
    
    config = OmegaConf.load(f"{opt.config}")
    model, global_step = load_model_from_config(config=config, ckpt=f"{opt.ckpt}", device=device)
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    one_time_load = {
        'model': model,
        'config': config,
        'sampler': sampler, 
        'opt': opt,
        'device': device
    }
    
    return one_time_load

one_time_load = get_model()

##### Frontend components #####

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon"),
)

mask_stroke_width = st.sidebar.slider("Mask stroke width: ", 10, 45, 40, step=5)
mask_color = '#ffffff'
sketch_stroke_width = st.sidebar.slider("Sketch stroke width: ", 2, 6, 3, step=1)
sketch_color = '#ffffff'
# colormap_stroke_width = st.sidebar.slider("Colormap stroke width: ", 15, 30, 20)
# colormap_color = st.sidebar.color_picker("Colormap hex: ", '#f39')

ref_type = st.sidebar.selectbox(
    'Select reference type:',
    ('rgb', 'cmap', 'cmap_with_edge', 'txt'))
scale = st.sidebar.slider("Scale: ", 0., 15., 10., step=0.5)
not_use_sketch_after = st.sidebar.slider("Detech: ", 0., 1., 1., step=0.25)

bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
exemplar_image = st.sidebar.file_uploader("Exemplar image:", type=["png", "jpg"])
bg_color = '#eee'
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Additional data for layout
canvas_mask_color = 'gray'
canvas_sketch_color = 'blue'
H, W = 512, 512
text_input = None

# Create a canvas component
col1, col2, col3, col4 = st.columns(4)
with col1:
    inject_color = st.button("Read Exemplar")
with col2:
    send_crop = st.button("Crop")
with col3:
    send_clicked = st.button("Inference")
with col4:
    send_export = st.button("Export")

grid = make_grid(cols=3, rows=2)

with grid[0][0]:
    canvas_result_mask = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=mask_stroke_width,
        stroke_color=mask_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        width=W,
        height=H,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

with grid[0][1]:
    canvas_result_sketch = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=sketch_stroke_width,
        stroke_color=sketch_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        width=W,
        height=H,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        key="full_app2",
    )

with grid[0][2]:
    if canvas_result_mask.image_data is not None or canvas_result_sketch.image_data is not None:
        if canvas_result_mask.image_data is not None and canvas_result_sketch.image_data is None:
            data = coloring(canvas_result_mask.image_data.copy(), canvas_mask_color)
            data = cv2.resize(data, (512, 512))
            st.image(data)
        if canvas_result_mask.image_data is None and canvas_result_sketch.image_data is not None:
            data = coloring(canvas_result_sketch.image_data.copy(), canvas_sketch_color)
            data = cv2.resize(data, (512, 512))
            st.image(data)
        if canvas_result_mask.image_data is not None and canvas_result_sketch.image_data is not None:
            colored_mask = coloring(canvas_result_mask.image_data.copy(), canvas_mask_color)
            colored_sketch = coloring(canvas_result_sketch.image_data.copy(), canvas_sketch_color)                
            data = np.clip((colored_mask + colored_sketch), 0, 255)
            data = cv2.resize(data, (512, 512))
            st.image(data)

### Reference path

ref_p_sketch_path = None
if inject_color:
    if ref_type in ['cmap', 'cmap_edge']:
        ref_p_sketch_path = preprocess_ref_canvas_bg(
            Image.fromarray(canvas_result_mask.image_data),
            Image.fromarray(canvas_result_sketch.image_data))
    elif ref_type in ['rgb']:
        ref_p_sketch_path = exemplar_image
        Image.open(ref_p_sketch_path).save('demo/ref_p.png')

with grid[1][0]:
    canvas_result_colormap = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#EA1010",        
        update_streamlit=realtime_update,
        background_image=Image.open(ref_p_sketch_path) if ref_p_sketch_path is not None else None,
        width=W,
        height=H,
        drawing_mode='rect',
        point_display_radius=0,
        key="full_app3",
    )

cropped_ref_p_path = None
if send_crop:
    obj = canvas_result_colormap.json_data["objects"][0]    
    top, left, width, height = obj['top'], obj['left'], obj['width'], obj['height']    
    img = Image.open('demo/ref_p.png').convert('RGB').resize((512,512))
    img = img.crop((left, top, left+width, top+height))
    img.save('demo/ref_p.png')
    cropped_ref_p_path = 'demo/ref_p.png'        


with grid[1][1]:
    canvas_ref = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#EA1010",        
        update_streamlit=realtime_update,
        background_image=Image.open(cropped_ref_p_path) if cropped_ref_p_path is not None else None,
        width=W,
        height=H,
        drawing_mode='rect',
        point_display_radius=0,
        key="full_app4",
    )


with grid[1][2]:
    if send_clicked:    
        one_time_load['opt']['ref_type'] == ref_type
        ret_image = paint_with_sketch(
            Image.open(bg_image), 
            Image.fromarray(canvas_result_mask.image_data),
            Image.fromarray(canvas_result_sketch.image_data),
            Image.fromarray(canvas_result_colormap.image_data),
            text_input,
            {'scale': scale, 'not_use_sketch_after': not_use_sketch_after},
            one_time_load)
        st.image(ret_image, caption='This is how your final image looks like 😉')

# When export button toggles
if send_export:
    print("Exporting the result")
    H, W = 512, 512
    img_p = Image.open('demo/img_p.png').convert('RGB').resize((H, W))    
    mask_np = cv2.imread('demo/mask.jpg')
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2RGB)
    sketch_np = cv2.imread('demo/sketch_p.png')
    sketch_np = cv2.cvtColor(sketch_np, cv2.COLOR_BGR2RGB)
    mask_np[sketch_np == 255] = 0
    mask_p = Image.fromarray(mask_np)
    overlayed = Image.blend(img_p, mask_p, 0.5)
    ref_p = Image.open('demo/ref_p.png').convert('RGB').resize((H, W))
    ret_img = Image.open('demo/ret_img.png').convert('RGB').resize((H, W))

    grid_img = Image.new('RGB', (W*4, H))
    for i, data in enumerate([img_p, overlayed, ref_p, ret_img]):
        grid_img.paste(data, (W*i, 0))        
    grid_img.save(f'demo/results/{str(datetime.now())}_grid_img.jpg')











