import os
from os import path
from typing import overload, Generator, Dict
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
import cv2
from omegaconf import OmegaConf

from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.bsrnet import RRDBNet
from model.scunet import SCUNet
from model.swinir import SwinIR
from utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from utils.face_restoration_helper import FaceRestoreHelper
from utils.helpers import (
    Pipeline,
    BSRNetPipeline, SwinIRPipeline, SCUNetPipeline,
    bicubic_resize
)
from utils.cond_fn import MSEGuidance, WeightedMSEGuidance

MAP_DIM = 8 # an image will be splitted into N x N patches

MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, but we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}

def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd

def paste_image_by_mask(img_a, img_b, img_b_mask):
    assert img_a.size == img_b.size
    assert img_b_mask.shape == (MAP_DIM,MAP_DIM)
    w, h = img_a.size

    upscaled_mask = cv2.resize((img_b_mask).astype(np.float32), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    image_A_np = np.array(img_a) # W x H x 3
    image_B_np = np.array(img_b)
    mask_3d = np.repeat(upscaled_mask[:, :, np.newaxis], 3, axis=2)
    result = image_B_np * mask_3d + image_A_np * (1 - mask_3d)
    final_image = Image.fromarray(result.astype(np.uint8))
    return final_image
    #############
    # w, h = img_a.size
    # patch_size = (w // MAP_DIM, h // MAP_DIM)
    # # Paste patches from image A to image B based on the binary mask
    # for i in range(MAP_DIM):
    #     for j in range(MAP_DIM):
    #         if img_b_mask[i, j] == 1:
    #             # Calculate the coordinates of the patch
    #             left = j * patch_size[0]
    #             upper = i * patch_size[1]
    #             right = left + patch_size[0]
    #             lower = upper + patch_size[1]
    #             # Crop the patch from image A
    #             patch = img_b.crop((left, upper, right, lower))
    #             # Paste the patch onto image B
    #             img_a.paste(patch, (left, upper))
    # return img_a

# mask is 512x512 with segmentation
def paste_image_by_mask_v2(img_a, img_b, img_b_mask):
    assert img_a.size == img_b.size
    image_A_np = np.array(img_a) # W x H x 3
    image_B_np = np.array(img_b)
    mask_3d = np.repeat(img_b_mask[:, :, np.newaxis], 3, axis=2)
    result = image_B_np * mask_3d + image_A_np * (1 - mask_3d)
    final_image = Image.fromarray(result.astype(np.uint8))
    return final_image

def initAllDiffusionModel(upscale=1):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.getcwd()
    # Change the current working directory to the script's directory
    os.chdir(script_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init stage 1
    bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    sd = load_model_from_url(MODELS["bsrnet"])
    bsrnet.load_state_dict(sd, strict=True)
    bsrnet.eval().to(device)
    
    # init stage 2
    ### load uent, vae, clip
    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    sd = load_model_from_url(MODELS["sd_v21"])
    unused = cldm.load_pretrained_sd(sd)
    print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
    ### load controlnet
    control_sd = load_model_from_url(MODELS["v2"])
    cldm.load_controlnet_from_ckpt(control_sd)
    print(f"strictly load controlnet weight")
    cldm.eval().to(device)
    ### load diffusion
    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)

    # init cond fn
    # g_scale = 0.0
    # g_start = 1001
    # g_stop = -1
    # g_space = "latent"
    # g_repeat = 1
    # cond_fn_cls = WeightedMSEGuidance
    # cond_fn = cond_fn_cls(
    #     scale=g_scale, t_start=g_start, t_stop=g_stop,
    #     space=g_space, repeat=g_repeat
    # )
    cond_fn = None

    # init pipeline
    pipeline_upscale = BSRNetPipeline(bsrnet, cldm, diffusion, cond_fn, device, upscale)
    sd = None
    control_sd = None
    torch.cuda.empty_cache()
    os.chdir(current_dir)
    return pipeline_upscale

def sr_pipeline_new(dataset, semantics_dir, output_dir, upscale=1, reproduce=True):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.getcwd()

    # Change the current working directory to the script's directory
    os.chdir(script_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loop_ctx = {}
    pipeline: Pipeline = None

    # init stage 1
    bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    sd = load_model_from_url(MODELS["bsrnet"])
    bsrnet.load_state_dict(sd, strict=True)
    bsrnet.eval().to(device)

    # init stage 2
    ### load uent, vae, clip
    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    sd = load_model_from_url(MODELS["sd_v21"])
    unused = cldm.load_pretrained_sd(sd)
    print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
    ### load controlnet
    control_sd = load_model_from_url(MODELS["v2"])
    cldm.load_controlnet_from_ckpt(control_sd)
    print(f"strictly load controlnet weight")
    cldm.eval().to(device)
    ### load diffusion
    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)

    # init cond fn
    # g_scale = 0.0
    # g_start = 1001
    # g_stop = -1
    # g_space = "latent"
    # g_repeat = 1
    # cond_fn_cls = WeightedMSEGuidance
    # cond_fn = cond_fn_cls(
    #     scale=g_scale, t_start=g_start, t_stop=g_stop,
    #     space=g_space, repeat=g_repeat
    # )
    cond_fn = None

    # init pipeline
    pipeline_upscale = BSRNetPipeline(bsrnet, cldm, diffusion, cond_fn, device, upscale)
    pipeline_fixed = BSRNetPipeline(bsrnet, cldm, diffusion, cond_fn, device, 1)

    # setup
    os.makedirs(output_dir, exist_ok=True)

    # negative prompt
    negative_prompt = "Blurry, Low Quality, Unnatural"
    positive_prompt = "Hyper Detail, Masterpiece, 4K, Realistic, Natural"
    if reproduce:
        negative_prompt = "Blurry, Low Quality"
        positive_prompt = "Hyper Detail, Masterpiece, 4K"

    # iterate all image files
    start_idx = 0
    end_idx = len(dataset)
    for i, data in enumerate(dataset[start_idx:end_idx]):
        # data = {
        #     'image_path': img_path,
        #     'image_name': img_id,
        #     'semantics': semantics
        # }
        # semantics = {
        #     "object": [],
        #     "detail": [],
        #     "summary": ""
        # }
        if not path.exists(data['image_path']):
            continue
        print(f"Image: {data['image_name']} ({start_idx+i+1}/{len(dataset)})")
        img = np.array(Image.open(data['image_path']).convert("RGB"))
        steps = data['diff_step']
        cfg = data['diff_cfg']
        
        object_map = None
        if reproduce:
            # load semantic map (clip patches method)
            object_map = np.load( path.join(semantics_dir, f"{data['image_name']}.dim-{MAP_DIM}.mask.t24.npy") )
        else:
            # load semantic map (clipseg method)
            object_map = np.load( path.join(semantics_dir, f"{data['image_name']}.mask.clipseg.npy") )
        
        # for upscale
        print(f"Upscaling by {upscale}X")
        temp, _ = pipeline_upscale.run(
            img[None], steps, 1.0, False,
            512, 256,
            f"{data['semantics']['summary']}, {positive_prompt}", negative_prompt, cfg,
            False
        )
        final_img = Image.fromarray(temp[0])
        final_img.save(path.join(output_dir, f"{data['image_name']}.0.png"))
        # final_img.save(path.join(output_dir, f"{data['image_name']}.final.png"))

        for obj_idx in range(len(data['semantics']['detail'][:1])):
            print(f"Adding detail of object {(obj_idx+1)}")
            mask = object_map[obj_idx]
            mask_img = Image.fromarray((mask*255).astype(np.uint8)).resize((512, 512))
            mask_img.save(path.join(output_dir, f"{data['image_name']}.obj{(obj_idx+1)}.mask.png"))
            temp, _ = pipeline_fixed.run(
                np.array(final_img)[None], steps, 1.0, False,
                512, 256,
                f"{data['semantics']['detail'][obj_idx]}, {positive_prompt}", negative_prompt, cfg,
                False
            )
            # temp, _ = pipeline_upscale.run(
            #     img[None], steps, 1.0, False,
            #     512, 256,
            #     f"{data['semantics']['object'][obj_idx]}, {positive_prompt}", negative_prompt, cfg/2,
            #     False
            # )
            tmp_img = Image.fromarray(temp[0])
            if reproduce:
                final_img = paste_image_by_mask(final_img, tmp_img, mask)
            else:
                final_img = paste_image_by_mask_v2(final_img, tmp_img, mask)
            final_img.save(path.join(output_dir, f"{data['image_name']}.obj{(obj_idx+1)}.png"))

        # print(f"Adding detail of summary")
        # temp, _ = pipeline_fixed.run(
        #     np.array(final_img)[None], steps, 1.0, False,
        #     512, 256,
        #     data['semantics']['summary'], negative_prompt, cfg,
        #     False
        # )
        # final_img = Image.fromarray(temp[0])
        # final_img.save(path.join(output_dir, f"{data['image_name']}.final.png"))

    bsrnet = None
    sd = None
    cldm = None
    diffusion = None
    torch.cuda.empty_cache()
    os.chdir(current_dir)

def sr_pipeline(dataset, semantics_dir, output_dir, steps=10, upscale=1, cfg=4, reproduce=True):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.getcwd()

    # Change the current working directory to the script's directory
    os.chdir(script_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loop_ctx = {}
    pipeline: Pipeline = None

    # init stage 1
    bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    sd = load_model_from_url(MODELS["bsrnet"])
    bsrnet.load_state_dict(sd, strict=True)
    bsrnet.eval().to(device)

    # init stage 2
    ### load uent, vae, clip
    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    sd = load_model_from_url(MODELS["sd_v21"])
    unused = cldm.load_pretrained_sd(sd)
    print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
    ### load controlnet
    control_sd = load_model_from_url(MODELS["v2"])
    cldm.load_controlnet_from_ckpt(control_sd)
    print(f"strictly load controlnet weight")
    cldm.eval().to(device)
    ### load diffusion
    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)

    # init cond fn
    # g_scale = 0.0
    # g_start = 1001
    # g_stop = -1
    # g_space = "latent"
    # g_repeat = 1
    # cond_fn_cls = WeightedMSEGuidance
    # cond_fn = cond_fn_cls(
    #     scale=g_scale, t_start=g_start, t_stop=g_stop,
    #     space=g_space, repeat=g_repeat
    # )
    cond_fn = None

    # init pipeline
    pipeline_upscale = BSRNetPipeline(bsrnet, cldm, diffusion, cond_fn, device, upscale)
    pipeline_fixed = BSRNetPipeline(bsrnet, cldm, diffusion, cond_fn, device, 1)

    # setup
    os.makedirs(output_dir, exist_ok=True)

    # negative prompt
    negative_prompt = "Blurry, Low Quality, Unnatural"
    positive_prompt = "Hyper Detail, Masterpiece, 4K, Realistic, Natural"
    if reproduce:
        negative_prompt = "Blurry, Low Quality"
        positive_prompt = "Hyper Detail, Masterpiece, 4K"

    # iterate all image files
    start_idx = 0
    end_idx = len(dataset)
    for i, data in enumerate(dataset[start_idx:end_idx]):
        # data = {
        #     'image_path': img_path,
        #     'image_name': img_id,
        #     'semantics': semantics
        # }
        # semantics = {
        #     "object": [],
        #     "detail": [],
        #     "summary": ""
        # }
        if not path.exists(data['image_path']):
            continue
        print(f"Image: {data['image_name']} ({start_idx+i+1}/{len(dataset)})")
        img = np.array(Image.open(data['image_path']).convert("RGB"))
        
        object_map = None
        if reproduce:
            # load semantic map (clip patches method)
            object_map = np.load( path.join(semantics_dir, f"{data['image_name']}.dim-{MAP_DIM}.mask.t24.npy") )
        else:
            # load semantic map (clipseg method)
            object_map = np.load( path.join(semantics_dir, f"{data['image_name']}.mask.clipseg.npy") )
        
        # for upscale
        print(f"Upscaling by {upscale}X")
        temp, _ = pipeline_upscale.run(
            img[None], steps, 1.0, False,
            512, 256,
            f"{data['semantics']['summary']}, {positive_prompt}", negative_prompt, cfg,
            False
        )
        final_img = Image.fromarray(temp[0])
        final_img.save(path.join(output_dir, f"{data['image_name']}.png"))
        # final_img.save(path.join(output_dir, f"{data['image_name']}.final.png"))

        # for obj_idx in range(len(data['semantics']['detail'])):
        #     print(f"Adding detail of object {(obj_idx+1)}")
        #     mask = object_map[obj_idx]
        #     mask_img = Image.fromarray((mask*255).astype(np.uint8)).resize((512, 512))
        #     mask_img.save(path.join(output_dir, f"{data['image_name']}.obj{(obj_idx+1)}.mask.png"))
        #     temp, _ = pipeline_fixed.run(
        #         np.array(final_img)[None], steps, 1.0, False,
        #         512, 256,
        #         f"{data['semantics']['object'][obj_idx]}, {positive_prompt}", negative_prompt, cfg,
        #         False
        #     )
        #     # temp, _ = pipeline_upscale.run(
        #     #     img[None], steps, 1.0, False,
        #     #     512, 256,
        #     #     f"{data['semantics']['object'][obj_idx]}, {positive_prompt}", negative_prompt, cfg/2,
        #     #     False
        #     # )
        #     tmp_img = Image.fromarray(temp[0])
        #     if reproduce:
        #         final_img = paste_image_by_mask(final_img, tmp_img, mask)
        #     else:
        #         final_img = paste_image_by_mask_v2(final_img, tmp_img, mask)
        #     final_img.save(path.join(output_dir, f"{data['image_name']}.obj{(obj_idx+1)}.png"))

        # print(f"Adding detail of summary")
        # temp, _ = pipeline_fixed.run(
        #     np.array(final_img)[None], steps, 1.0, False,
        #     512, 256,
        #     data['semantics']['summary'], negative_prompt, cfg,
        #     False
        # )
        # final_img = Image.fromarray(temp[0])
        # final_img.save(path.join(output_dir, f"{data['image_name']}.final.png"))

    bsrnet = None
    sd = None
    cldm = None
    diffusion = None
    torch.cuda.empty_cache()
    os.chdir(current_dir)
    



    
    
    


















    