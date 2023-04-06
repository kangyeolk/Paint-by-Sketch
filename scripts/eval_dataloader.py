import argparse, os, sys, glob

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision

sys.path.insert(0, '/home/nas2_userF/kangyeol/Project/webtoon2022/Paint-by-Example')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize

wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

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


"""
CUDA_VISIBLE_DEVICES=1 python scripts/eval_dataloader.py \
--plms --outdir results/cmap_naive \
--ckpt Cartoon_v1_figure_cmap_naive \
--seed 321 \
--scale 5 \
--gpu 1 \
--num_samples 200

CUDA_VISIBLE_DEVICES=2 python scripts/eval_dataloader.py \
--plms --outdir results/cmap_naive_part \
--ckpt Cartoon_v1_cmap_naive_part \
--seed 321 \
--scale 5 \
--gpu 3 \
--num_samples 200

CUDA_VISIBLE_DEVICES=0 python scripts/eval_dataloader.py \
--plms --outdir results/Cartoon_v1_aesthetic_image \
--ckpt Cartoon_v1_aesthetic_image \
--seed 321 \
--scale 5 \
--gpu 0 \
--n_samples 8 \
--num_iterations=1
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="gpu id to use",
    )
    parser.add_argument(
        "--num_slice",
        type=int,
        default=100,
        help="Number of batch to use",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of iterations to forward",
    )
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default='replace',
        choices=['replace', 'latent_fusion']
    )


    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    log_ret = os.path.join('models', opt.ckpt)
    run_name = sorted(os.listdir(log_ret))[-1]
    config_name = [c for c in os.listdir(os.path.join(log_ret, run_name, 'configs')) if 'project' in c][0]
    
    # Overwrite ckpt, config
    opt.ckpt = os.path.join(log_ret, run_name, 'checkpoints', 'last.ckpt')
    opt.config = os.path.join(log_ret, run_name, 'configs', config_name)
        
    
    # GPU
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(f"{opt.config}")
    
    # Dataset        
    batch_size = opt.n_samples
    config.data.params['batch_size'] = batch_size    
    ref_type = config.model.params['cond_stage_key']
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    train_loader = data._train_dataloader()
    val_loader = data._val_dataloader()
    
    model, global_step = load_model_from_config(config=config, ckpt=f"{opt.ckpt}", device=device)
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    outpath = os.path.join(outpath, str(global_step).zfill(7))
    os.makedirs(outpath, exist_ok=True)
    
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        

    def get_input_sketch(config, batch, mask):
        sketch_stage_config = config.model['params']['sketch_stage_config']
        if sketch_stage_config['input_edge_type'] == 'full':           
            in_sketch = batch['sketch']
        elif sketch_stage_config['input_edge_type'] == 'only_mask_region':           
            in_sketch = ((batch['sketch']+1)/2) * (1 - mask)
            in_sketch = 2*in_sketch-1
        elif sketch_stage_config['input_edge_type'] == 'erase_mask_region':           
            in_sketch = ((batch['sketch']+1)/2) * mask
            in_sketch = 2*in_sketch-1
        else:
            in_sketch = None 
        return in_sketch

    def un_norm(x):
        return (x+1.0)/2.0
    
    def un_norm_clip(x):
        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
        return x

    def get_tensor_clip(normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def mask_to_bbox(mask_np: np.array):
        # NOTE: reverse coordinate
        # mask_np: (bsz, 1, 512, 512)
        bboxes = []
        for data in mask_np:
            data = data[0, :, :]
            H, W = data.shape
            xs, ys = np.where(data==0) # mask region value: 0 
            max_x = min(xs.max()+10, W)
            min_x = max(0, xs.min()-10)
            max_y = min(ys.max()+10, H)
            min_y = max(0, ys.min()-10)
            bbox = [min_x, min_y, max_x, max_y]
            bboxes.append(bbox)
        return bboxes

    def gen_new_ref_tensor(x_imgs, bboxes):        
        ret = []
        for x_img, bbox in zip(x_imgs, bboxes):
            img_p_np = np.array(x_img)
            ref_image_tensor=img_p_np[bbox[0]:bbox[2],bbox[1]:bbox[3],:]
            ref_image_tensor=Image.fromarray(ref_image_tensor)
            ref_image_tensor=ref_image_tensor.resize((224,224))
            ref_image_tensor=get_tensor_clip()(ref_image_tensor)
            ret.append(ref_image_tensor)
        ret = torch.stack(ret, dim=0)
        return ret
    
    def log_txt_as_img(wh, xc, size=10):
        # wh a tuple of (width, height)
        # xc a list of captions to plot
        b = len(xc)
        txts = list()
        for bi in range(b):
            txt = Image.new("RGB", wh, color="white")
            draw = ImageDraw.Draw(txt)
            # font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
            nc = int(40 * (wh[0] / 256))
            lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))
            try:
                draw.text((0, 0), lines, fill="black")
            except UnicodeEncodeError:
                print("Cant encode string for logging. Skipping.")

            txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
            txts.append(txt)
        txts = np.stack(txts)
        txts = torch.tensor(txts)
        return txts

    
    # def chunks(lst, n):
    #     """Yield successive n-sized chunks from lst."""
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]

    # if opt.split_cfg is not None:
    #     seed, num_div = opt.split_cfg.split('-')
    #     per_seed_data = len(val_loader.dataset) // num_div                
    #     per_seed_data = range(len(val_loader.dataset))        

    # splited_list = list(chunks(files, N))
    # splited_list[-2] = splited_list[-2] + splited_list[-1]
    # splited_list.pop(-1)    
    # assert len(splited_list) == args.max_split, f'{len(splited_list)}'        
    # TODO: Fusion strategy

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    image_tensor = batch['GT']
                    inpaint_image = batch['inpaint_image']
                    mask_tensor = batch['inpaint_mask']
                    in_sketch = get_input_sketch(config, batch, mask_tensor)
                    
                    test_model_kwargs={}
                    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
                    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
                    test_model_kwargs['in_sketch'] = None if in_sketch is None else in_sketch.to(device)
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])
                    if in_sketch is not None:
                        test_model_kwargs['in_sketch']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['in_sketch'])
                    if ref_type == 'image':
                        ref_tensor = batch['ref_imgs']
                        ref_tensor = ref_tensor.to(device)
                    elif ref_type == 'txt':
                        ref_tensor = batch['txt']

                    ## Additional vars to control sketch things..
                    # test_model_kwargs['sketch_config'] = {}
                    # test_model_kwargs['sketch_config']['not_use_sketch_after'] = 1.0 # 0.5, 0.75, 1.0
                    
                    for iter in range(opt.num_iterations):
                        
                        # Forwarding
                        uc = None
                        if ref_type == 'image':
                            if opt.scale != 1.0:
                                uc = model.learnable_vector                    
                                uc = uc.repeat(opt.n_samples, 1, 1)
                            c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                            c = model.proj_out(c)
                        elif ref_type == 'txt':
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(opt.n_samples * [""])
                            if isinstance(ref_tensor, tuple):
                                ref_tensor = list(ref_tensor)           
                            c = model.get_learned_conditioning(ref_tensor)
                        
                        # if opt.scale != 1.0:
                        #     uc = model.learnable_vector
                        #     uc = uc.repeat(opt.n_samples, 1, 1)
                        # c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                        # c = model.proj_out(c)
                        # import pdb; pdb.set_trace()
                    
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            test_model_kwargs=test_model_kwargs)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        
                        if ref_type == 'image':
                            ref_img=ref_tensor
                            ref_img=Resize([512,512])(ref_img)
                        
                        for i,x_sample in enumerate(x_checked_image_torch):
                            filename = f'{idx}'.zfill(6) + f'{i}'.zfill(2) + '_' + f'{iter}'.zfill(2)
                            all_img=[]
                            all_img.append(un_norm(image_tensor[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            if in_sketch is not None:
                                all_img.append(un_norm(in_sketch[i]).cpu())
                            if ref_type == 'image':
                                all_img.append(un_norm_clip(ref_img[i]).cpu())
                            elif ref_type == 'txt':
                                all_img.append(log_txt_as_img((512,512), [ref_tensor[i]])[0])
                            all_img.append(x_sample)                            
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(outpath, 'grid-'+filename+'_'+str(opt.seed)+'.png'))

                        # Generate new reference image
                        if (iter + 1) <= opt.num_iterations:
                            bboxes = mask_to_bbox(255*mask_tensor.detach().cpu().numpy())
                            x_imgs = []
                            for x_sample_ddim in x_samples_ddim:
                                x_imgs.append(
                                    Image.fromarray((255*x_sample_ddim).astype(np.uint8)))                            
                            ref_tensor = gen_new_ref_tensor(x_imgs, bboxes)
                            ref_tensor = ref_tensor.to(device)

                        if (idx + 1) % opt.num_slice == 0:
                            break

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
