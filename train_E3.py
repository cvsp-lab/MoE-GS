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

import os
import json
import sys
import uuid
import math
from random import randint
import gc

import torch
from tqdm import tqdm
from PIL import Image
import joblib

from gaussian_renderer import render_E3 as render
from scene import Scene, getmodel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelEmbParams, ModelHiddenParams
import torchvision
import torch.nn.functional as F
import imageio
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False
torch.set_default_dtype(torch.float32)
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
import numpy as np

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
def training(dataset, emb, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    """
    Main training routine for the MoE-GS framework.

    This function initializes and loads multiple expert models (Ex4DGS, E-D3DGS, 
    and 4DGaussians), prepares their associated Gaussian and Scene 
    instances, and sets up optimizers for training expert-specific weights.

    Args:
    dataset: Dataset configuration and metadata.
    emb: Hyperparameters for E-D3DGS models.
    hyper: Hyperparameters for 4DGaussians.
    opt: Training options and optimizer settings.
    pipe: Rendering pipeline configuration.
    testing_iterations: Iterations at which to run evaluation.
    saving_iterations: Iterations at which to save checkpoints.
    checkpoint_iterations: Iterations at which to resume checkpoints.
    checkpoint: Optional pre-trained checkpoint to resume from.
    debug_from: Iteration index to enable debug mode.
    args: Command-line or configuration arguments.

    Returns:
        None. This function orchestrates the initialization of all expert models 
        and their optimizers for subsequent training loops.
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)

    # Load Gaussian Models & Scenes for Each Expert
    """
    In the 3-expert configuration, we use:
        - Ex4DGS
        - E-D3DGS
        - 4DGaussians (CVPR)
    """

    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    gaussians = GaussianModel(dataset.sh_degree, dataset.start_duration, dataset.time_interval, dataset.time_pad, 
                              interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type, 
                              time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, 
                              kernel_size=dataset.kernel_size, rgbfunction=args.rgbfunction
    )
    scene = Scene(
        dataset, gaussians, args.load_iteration, use_timepad=True, 
        save_path = args.save_path, init_mixprob = args.init_mixprob
    )
    args.duration = dataset.duration
    cameras_extent = scene.cameras_extent
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Load E-D3DGS model (Canonical Gaussians + Deformation Network)
    gaussians_e = EGaussianModel(dataset.sh_degree, emb, rgbfunction=args.rgbfunction)
    scene_e =  EScene(
        dataset, gaussians_e, load_iteration = args.load_iteration, 
        loader=dataset.loader, duration=emb.total_num_frames, opt=opt, 
        cameras_extent=cameras_extent, save_path = args.save_path, init_mixprob = args.init_mixprob
    )

    # Load 4D Gaussians (CVPR'24) model
    gaussians_h = HGaussianModel(dataset.sh_degree, hyper, rgbfunction=args.rgbfunction)
    scene_h = HScene(
        dataset, gaussians_h, args.load_iteration, load_coarse=None, 
        cameras_extent=cameras_extent, save_path = args.save_path, init_mixprob = args.init_mixprob
    )

    # Setup Training for All Experts
    gaussians.training_setup(opt)
    gaussians_e.training_setup(opt)
    gaussians_h.training_setup(opt)

    saving_iterations = args.save_iterations
    testing_iterations = args.test_iterations
        
    with open(os.path.join(args.save_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    train_images = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # Main Training Loop
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # Update Learning Rates for Each Expert
        gaussians.update_learning_rate(iteration)
        gaussians_e.update_learning_rate(iteration)
        gaussians_h.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack, train_images = scene.getTrainCameras(return_as='generator', shuffle=True)
            viewpoint_stack = viewpoint_stack.copy()
                        
        viewpoint_cam = viewpoint_stack.pop(0)
        gt_image = next(train_images).cuda()
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        cam_no = viewpoint_cam.cam_no
        render_pkg = render(
            viewpoint_cam, gaussians, gaussians_e, gaussians_h, 
            pipe, bg, near=dataset.near, 
            far=dataset.far,iter=scene_e.pretrained_iter, 
            num_down_emb_c=emb.min_embeddings, 
            num_down_emb_f=emb.min_embeddings, 
            cam_no=cam_no)
        
        # Unpack rendered outputs
        image, e_image, h_image, = (
            render_pkg["render"], render_pkg["e_render"], render_pkg["h_render"],
        )
        weight, e_weight, h_weight = (
            render_pkg["weight"], render_pkg["e_weight"], render_pkg["h_weight"],
        )
        viewspace_point_tensor, viewspace_point_error_tensor, visibility_filter, radii, depth, flow, acc, idxs = (
             render_pkg["viewspace_points"], render_pkg["viewspace_l1points"], render_pkg["visibility_filter"],    
            render_pkg["radii"], render_pkg["depth"], render_pkg["opticalflow"], render_pkg["acc"], render_pkg["dominent_idxs"] 

        )
        
        # Pixel-wise MoE fusion
        stacked_weight = F.softmax(
            torch.stack([weight, e_weight, h_weight], dim=0) / args.temperature, dim=0
        )
        weight, e_weight, h_weight = (
            stacked_weight[0].unsqueeze(0), 
            stacked_weight[1].unsqueeze(0), 
            stacked_weight[2].unsqueeze(0), 
        )
        moe_image = (
            (image * weight)
            + (e_image * e_weight)
            + (h_image * h_weight)
        )

        # Loss
        Ll1 = l1_loss(moe_image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(moe_image, gt_image)) 

        loss.backward()
        iter_end.record()
        gaussians.mark_error(loss.item(), viewpoint_cam.timestamp)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()
            psnr_log = psnr(moe_image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_log:.{2}f}", "Loss": f"{ema_loss_for_log:.{6}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, scene_e, scene_h, pipe, bg, dataset.near, dataset.far, emb, args.save_path)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians models...".format(iteration))

                scene.save(iteration, args.save_path)
                scene_e.save(iteration, args.save_path)
                scene_h.save(iteration, args.save_path)
                stg_fh_path = os.path.join(args.save_path, "STG", '0to149')
                stg_sh_path = os.path.join(args.save_path, "STG", '150to299')
            
            if iteration < opt.iterations:
                # Apply optimizer step and clear gradients
                for model in [
                    gaussians, gaussians_e, gaussians_h
                ]:
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                
                torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    # if not args.model_path:
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str=os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.save_path))
    os.makedirs(args.save_path, exist_ok = True)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.save_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, scene_e : Scene, scene_h : Scene, pipe, bg, near, far, emb, save_path):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    name = 'test'
   # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        render_path = os.path.join(save_path,'test',"itrs_{}".format(iteration))
        os.makedirs(os.path.join(render_path, "videos"), exist_ok=True)

        test_viewpoint_stack, test_images = scene.getTestCameras(
            return_as='generator', shuffle=False, n_job=1
        )
        test_viewpoint_stack = test_viewpoint_stack.copy()

        moe_images, moe_psnrs, moe_ssims, moe_lpipss = [], [], [], []
        weights, e_weights, h_weights = [], [], []

        images, e_images, h_images = [], [], []

        for idx, viewpoint in enumerate(tqdm(test_viewpoint_stack)):
            gt = next(test_images).cuda()

            render_pkg = render(
                viewpoint, scene.gaussians, scene_e.gaussians, scene_h.gaussians, 
                pipe, bg, near=near, 
                far=far,iter=scene_e.pretrained_iter, 
                num_down_emb_c=emb.min_embeddings, 
                num_down_emb_f=emb.min_embeddings, cam_no=None,
            )

            # Unpack rendered outputs
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

            # Collect rendered frames and metrics
            moe_images.append(to8b(moe_render).transpose(1, 2, 0))
            
            images.append(to8b(image).transpose(1, 2, 0))
            e_images.append(to8b(e_image).transpose(1, 2, 0))
            h_images.append(to8b(h_image).transpose(1, 2, 0))

            moe_psnrs.append(psnr(moe_render.unsqueeze(0), gt.unsqueeze(0)).mean().item())
            # moe_ssims.append(ssim(moe_render.unsqueeze(0), gt.unsqueeze(0))) 
            # moe_lpipss.append(lpips(moe_render.unsqueeze(0), gt.unsqueeze(0), net_type='alex')) #

            # Collect visualization gates
            weights.append(to8b(weight).transpose(1, 2, 0)[..., 0])
            e_weights.append(to8b(e_weight).transpose(1, 2, 0)[..., 0])
            h_weights.append(to8b(h_weight).transpose(1, 2, 0))
            torch.cuda.empty_cache()

        imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_MoE.mp4'), moe_images, fps=30)

        imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_Ex4DGS.mp4'), weights, fps=30)
        imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_E-D3DGS.mp4'), e_weights, fps=30)
        imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'weights_4DGaussians.mp4'), h_weights, fps=30)
        
        # imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_Ex4DGS.mp4'), images, fps=30)
        # imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_E-D3DGS.mp4'), e_images, fps=30)
        # imageio.mimwrite(os.path.join(save_path, name, "itrs_{}".format(iteration), "videos", 'video_4DGaussians.mp4'), h_images, fps=30)

        mean_results = {
            "MoE-GS_PSNR": torch.tensor(moe_psnrs).mean().item(),
            # "MoE-GS_SSIM": torch.tensor(moe_ssims).mean().item(),
            # "MoE-GS_LPIPS": torch.tensor(moe_lpipss).mean().item(),
            }
        
        with open(save_path + '/' + 'test' + "/itrs_{}".format(iteration) + "/" + "mean_metrics.json", 'w') as fp:
            json.dump(mean_results, fp, indent=True)

        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ep = ModelEmbParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 2000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 7000_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 2000, 5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configpath", type=str, default = "None")
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

    parser.add_argument("--init_mixprob", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.01)

    args = parser.parse_args(sys.argv[1:])

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.test_iterations)
    print("Optimizing " + args.model_path)

    # Load Main Config (Required)
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    # Load Deformation Config
    import mmcv
    from thirdparty.embedding.utils.params_utils import merge_hparams
    emb_cfg_path = f"thirdparty/embedding/arguments/dynerf/{os.path.basename(args.source_path)}.py"
    if not os.path.exists(emb_cfg_path):
        raise FileNotFoundError(f"Missing E-D3DGS config: {emb_cfg_path}")
    emb_config = mmcv.Config.fromfile(emb_cfg_path)
    args = merge_hparams(args, emb_config)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), ep.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
