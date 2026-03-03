#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import gc
import os
from os import makedirs
import json
import time
import sys
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import torch
import torchvision
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim
from lpipsPyTorch import lpips
import joblib

from scene import Scene
from gaussian_renderer import render_E3 as render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelEmbParams, ModelHiddenParams, get_combined_args

from scene.c_gaussian_model import CGaussianModel as GaussianModel
from utils.loss_utils import ssim
import numpy as np
import imageio
import torch.nn.functional as F
from collections import defaultdict

""" Import Experts """
# E-D3DGS
from thirdparty.embedding.scene.gaussian_model import EGaussianModel
from thirdparty.embedding.scene import EScene

# 4d-gaussian-splatting
# from thirdparty.reparmetrize.scene.gaussian_model import FGaussianModel
# from thirdparty.reparmetrize.scene import FScene

from thirdparty.polynomial.scene import SScene 
from thirdparty.polynomial.scene.oursfull import SGaussianModel
from thirdparty.polynomial.helper_train import trbfunction

# 4DGaussians
from thirdparty.Hexplane.scene.gaussian_model import HGaussianModel
from thirdparty.Hexplane.scene import HScene

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(save_path, name, iteration, scene, gaussians, gaussians_e, gaussians_h, pipeline, background, inverval=1, near=0.2, far=100.0, save_img=False, embparam=None, E_iter = None):
    gts_path = os.path.join(save_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(save_path, name, "ours_{}".format(iteration), "renders")

    if save_img:
        makedirs(gts_path, exist_ok=True)
        makedirs(render_path, exist_ok=True)

    if name == "train":
        viewpoint_stack, images = scene.getTrainCameras(return_as='generator', shuffle=False)
        viewpoint_stack = viewpoint_stack.copy()
        
    else:
        viewpoint_stack, images = scene.getTestCameras(return_as='generator', shuffle=False)
        viewpoint_stack = viewpoint_stack.copy()
    
    num_down_emb_c = embparam.min_embeddings
    num_down_emb_f = embparam.min_embeddings    

    idx = 0
    count = 0
    psnr_sum = 0

    moe_images, moe_psnrs, moe_ssims, moe_lpipss, moe_times = [], [], [], [], []

    weights, e_weights, h_weights = [], [], []
    i_images, e_images, h_images = [], [], []

    rbfbasefunction = trbfunction

    while(len(viewpoint_stack)):
        cam = viewpoint_stack.pop(0)
        gt = next(images).cuda()
        cam_no = cam.cam_no

        if idx % inverval == 0:
            render_pkg = render(
                cam, gaussians, gaussians_e, gaussians_h, 
                pipeline, background, near=near, 
                far=far,iter=E_iter, 
                num_down_emb_c=num_down_emb_c, 
                num_down_emb_f=num_down_emb_f, cam_no=cam_no)

            image, e_image, h_image, = (
                render_pkg["render"], render_pkg["e_render"], render_pkg["h_render"],
            )
            weight, e_weight, h_weight = (
                render_pkg["weight"], render_pkg["e_weight"], render_pkg["h_weight"],
            )

        # Pixel-wise MoE fusion
            stacked_weight = F.softmax(
                torch.stack([weight, e_weight, h_weight], dim=0), dim=0
            )
            weight, e_weight, h_weight = (
                stacked_weight[0].unsqueeze(0), 
                stacked_weight[1].unsqueeze(0), 
                stacked_weight[2].unsqueeze(0), 
            )
            moe_render = (
                (image * weight)
                + (e_image * e_weight)
                + (h_image * h_weight)
            )
            

            moe_images.append(to8b(moe_render).transpose(1, 2, 0))

            i_images.append(to8b(image).transpose(1, 2, 0))
            e_images.append(to8b(e_image).transpose(1, 2, 0))
            h_images.append(to8b(h_image).transpose(1, 2, 0))

            weights.append(to8b(weight).transpose(1, 2, 0)[..., 0])
            e_weights.append(to8b(e_weight).transpose(1, 2, 0)[..., 0])
            h_weights.append(to8b(h_weight).transpose(1, 2, 0))

            moe_psnrs.append(psnr(moe_render.unsqueeze(0), gt.unsqueeze(0)).mean().item())
            moe_ssims.append(ssim(moe_render.unsqueeze(0), gt.unsqueeze(0))) 
            moe_lpipss.append(lpips(moe_render.unsqueeze(0), gt.unsqueeze(0), net_type='alex')) #

            torch.cuda.empty_cache()

            count += 1
        idx += 1
        
    
    # start timing
    for _ in range(5):
        for idx in range(500):
            st = time.time()
            render_pkg = render(
                cam, gaussians, gaussians_e, gaussians_h, 
                pipeline, background, near=near, 
                far=far,iter=E_iter, 
                num_down_emb_c=num_down_emb_c, 
                num_down_emb_f=num_down_emb_f, cam_no=cam_no)

            if idx > 100: #warm up
                moe_times.append(time.time() - st)

    

    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_MoE.mp4'), moe_images, fps=30)

    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_Ex4DGS.mp4'), weights, fps=30)
    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_E-D3DGS.mp4'), e_weights, fps=30)
    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_4DGaussians.mp4'), h_weights, fps=30)
    
    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_Ex4DGS.mp4'), i_images, fps=30)
    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_E-D3DGS.mp4'), e_images, fps=30)
    imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_4DGaussians.mp4'), h_images, fps=30)

        
    mean_results = {
        "MoE-GS_PSNR": torch.tensor(moe_psnrs).mean().item(),
        "MoE-GS_SSIM": torch.tensor(moe_ssims).mean().item(),
        "MoE-GS_LPIPS": torch.tensor(moe_lpipss).mean().item(),
        "MoE-GS_times": torch.tensor(moe_times).mean().item(),
        }

    with open(save_path + "/" + "mean_metrics.json", 'w') as fp:
        json.dump(mean_results, fp, indent=True)

    torch.cuda.empty_cache()

    
def render_sets(dataset : ModelParams, embparam, hyperparam, iteration : int, opt : OptimizationParams, pipeline : PipelineParams, skip_train : bool, train_inverval : int, skip_test : bool, save_img : bool, save_path : str, \
                gaussian_dim : int, time_duration : float, num_pts : int, num_pts_ratio : float, rot_4d : bool, force_sh_3d : bool, batch_size : int):
    """
    Unified rendering entry for all experts in MoE-GS (E4 version).

    This function loads pretrained models (Ex4DGS, E-D3DGS, STG, 4DGaussians) w/ Per-Gaussian Weights,
    restores checkpoints, and runs rendering for both train/test splits.
    """
    with torch.no_grad():
        # --- Ex4DGS ---
        gaussians = GaussianModel(
            dataset.sh_degree, dataset.duration, dataset.time_interval, 
            dataset.time_pad, interp_type=dataset.interp_type, 
            time_pad_type=dataset.time_pad_type, rgbfunction=args.rgbfunction
        )
        scene = Scene(
            dataset, gaussians, load_iteration=iteration, 
            shuffle=False, opt=opt, save_path=save_path, render=True
        )

        # --- E-D3DGS ---
        gaussians_e = EGaussianModel(dataset.sh_degree, embparam)
        scene_e =  EScene(
            dataset, gaussians_e, load_iteration = iteration, 
            loader=dataset.loader, duration=embparam.total_num_frames, 
            opt=opt, save_path=save_path, render=True
        )

        # --- 4DGaussians ---
        gaussians_h = HGaussianModel(dataset.sh_degree, hyperparam)
        scene_h = HScene(dataset, gaussians_h, load_iteration=iteration, load_coarse=None, render=True, save_path=save_path)

        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            print(f"[INFO] Rendering train viewpoints ...")
            render_set(save_path, "train", scene.loaded_iter, scene, gaussians, gaussians_e, gaussians_h, pipeline, background, near=dataset.near, far=dataset.far, save_img=save_img, embparam=embparam, E_iter = scene_e.pretrained_iter)

        if not skip_test:
            print(f"[INFO] Rendering test viewpoints ...")
            render_set(save_path, "test", scene.loaded_iter, scene, gaussians, gaussians_e, gaussians_h, pipeline, background, near=dataset.near, far=dataset.far, save_img=save_img, embparam=embparam, E_iter = scene_e.pretrained_iter)
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    emb = ModelEmbParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--train_inverval", default=1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument('--load_iteration', type=int, default=-1)
    parser.add_argument("--save_path", type=str, default = None)
    parser.add_argument("--rgbfunction", type=str, default = "sandwichlite")
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0.0, 10.0])
    parser.add_argument('--num_pts', type=int, default=300_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", type=bool, default=True)
    parser.add_argument("--force_sh_3d", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)

    args = get_combined_args(parser)
    print(f"[INFO] Rendering model from: {args.save_path}")
    
    # Embedding Config
    import mmcv
    from thirdparty.embedding.utils.params_utils import merge_hparams
    emb_cfg_path = f"thirdparty/embedding/arguments/dynerf/{os.path.basename(args.source_path)}.py"
    if not os.path.exists(emb_cfg_path):
        raise FileNotFoundError(f"Missing E-D3DGS config: {emb_cfg_path}")
    emb_config = mmcv.Config.fromfile(emb_cfg_path)
    args = merge_hparams(args, emb_config)

    # Initialize system state (RNG)
    render_sets(model.extract(args), emb.extract(args), hp.extract(args), args.iteration, opt.extract(args), pipeline.extract(args), args.skip_train, args.train_inverval, args.skip_test, args.save_img, args.save_path, \
                args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)