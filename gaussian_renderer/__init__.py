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
import math

import torch
from torch.nn import functional as F
from diff_gaussian_rasterization_df import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

# E-D3DGS rasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as EGaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer as  EGaussianRasterizer

from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings as LGRsetting 
from diff_gaussian_rasterization_ch3 import GaussianRasterizer as LGRzer


def freeze_tensors(*tensors):
    return [t.detach() for t in tensors]

def render_ex4dgs(pc, cam, screenspace_points, rasterizer, m_rasterizer, timestamp, pointtimes, pointtimes_motion, mode, training):
    means3D = pc.get_xyz_at_t(timestamp, mode=mode, training=training)
    means2D = screenspace_points
    opacity = pc.get_opacity_at_t(timestamp, mode=mode, training=training)
    scales = pc.get_scaling(mode=mode)
    rotations = pc.get_rotation_at_t(timestamp, mode=mode)
    shs = pc.get_features(mode=mode)

    flow = torch.zeros_like(means3D, requires_grad=True) + 0
    # try:
    #     flow.retain_grad()
    # except:
    #     pass

    # [Freeze pretrained model parameters]
    #   - These are from pretrained Ex4DGS model → no gradient update
    #   - Only mixture / routing weights will be updated
    means3D, means2D, shs, scales, rotations, opacity = \
            freeze_tensors(means3D, means2D, shs, scales, rotations, opacity)

    rendered_image, radii, rendered_depth, out_flow, acc, idxs = rasterizer(
        means3D=means3D, means2D=means2D, dir3D = flow, shs=shs, opacities=opacity,
        colors_precomp = None, scales=scales, rotations=rotations, cov3D_precomp = None
    )

    trbfdistanceoffset = cam.emb_timestamp*pointtimes
    trbfdistanceoffset_motion = cam.emb_timestamp*pointtimes_motion
    spacetime_mask = pc.get_motion_mask(trbfdistanceoffset.detach(), trbfdistanceoffset_motion.detach())

    rendered_weight, _, _ = m_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = spacetime_mask,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    return dict(
        image=rendered_image,
        radii=radii,
        depth=rendered_depth,
        out_flow=out_flow,
        flow=flow,
        acc=acc,
        idxs=idxs,
        weight = rendered_weight,
        screenspace_points = screenspace_points,
    )

def render_stg(pc, cam, screenspace_points, pointtimes, rasterizer, w_rasterizer, timestamp, basicfunction):
    means3D = pc.get_xyz 
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)

    # temporal opacity를 detach
    opacity = pointopacity * trbfoutput.detach()  # - 0.5
    pc.trbfoutput = trbfoutput

    scales = pc.get_scaling
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D + pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    shs = pc.get_features()

    flow = torch.zeros_like(means3D, requires_grad=True) + 0
    # try:
    #     flow.retain_grad()
    # except:
    #     pass

    means3D = means3D.detach()
    means2D = means2D.detach()
    shs = shs.detach()
    scales = scales.detach()
    rotations = rotations.detach()
    opacity = opacity.detach()

    means3D, means2D, shs, scales, rotations, opacity = \
            freeze_tensors(means3D, means2D, shs, scales, rotations, opacity)

    cov3D_precomp = None
    rendered_image, radii, rendered_depth, out_flow, acc, idxs = rasterizer(
        means3D=means3D, means2D=means2D, dir3D = flow, shs=shs, opacities=opacity,
        colors_precomp = None, scales=scales, rotations=rotations, cov3D_precomp = None
    )

    trbfdistanceoffset = cam.emb_timestamp*pointtimes
    spacetime_mask = pc.get_motion_mask(trbfdistanceoffset.detach())

    rendered_weight, _, _ = w_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = spacetime_mask,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return dict(
        image=rendered_image,
        radii=radii,
        depth=rendered_depth,
        flow=out_flow,
        acc=acc,
        idxs=idxs,
        weight = rendered_weight
    )

def render_ed3dgs(pc, cam, screenspace_points, rasterizer, w_rasterizer, pointtimes, cam_no, iter, num_down_emb_c, num_down_emb_f):
    means3D = pc.get_xyz
    time = torch.tensor(cam.emb_timestamp).to(means3D.device).repeat(means3D.shape[0],1)
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = pc._scaling
    rotations = pc._rotation

    means3D_final, scales_final, rotations_final, opacity_final, shs_final, extras = pc._deformation(means3D, scales, 
        rotations, opacity, time, cam_no, pc, None, shs, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    depth = None
    colors_precomp = None
    cov3D_precomp = None

    # [Freeze pretrained model parameters]
    #   - These are from pretrained E-D3DGS model → no gradient update
    #   - Only mixture / routing weights will be updated
    means3D_final, means2D, shs_final, scales_final, rotations_final, opacity_final = \
            freeze_tensors(means3D_final, means2D, shs_final, scales_final, rotations_final, opacity_final)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    outputs = rasterizer(
        means3D = means3D_final, means2D = means2D, shs = shs_final,
        colors_precomp = colors_precomp, opacities = opacity_final, 
        scales = scales_final, rotations = rotations_final, cov3D_precomp = cov3D_precomp
    )

    if len(outputs) == 2:
        rendered_image, radii = outputs
    elif len(outputs) == 3:
        rendered_image, radii, depth = outputs

    trbfdistanceoffset = cam.emb_timestamp*pointtimes
    spacetime_mask = pc.get_motion_mask(trbfdistanceoffset.detach())

    rendered_weight, _, _ = w_rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = spacetime_mask,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    return dict(
        image=rendered_image,
        radii=radii,
        depth=depth,
        weight = rendered_weight
    )


def render_4dgaussians(pc, cam, screenspace_points, rasterizer, w_rasterizer, pointtimes):
    means3D = pc.get_xyz
    time = torch.tensor(cam.emb_timestamp).to(means3D.device).repeat(means3D.shape[0],1)
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = pc._scaling
    rotations = pc._rotation

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                            rotations, opacity, shs,
                                                            time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    # [Freeze pretrained model parameters]
    #   - These are from pretrained 4DGaussians model → no gradient update
    #   - Only mixture / routing weights will be updated
    means3D_final, means2D, shs_final, scales_final, rotations_final, opacity_final = \
            freeze_tensors(means3D_final, means2D, shs_final, scales_final, rotations_final, opacity_final)

    colors_precomp  = None
    cov3D_precomp = None
    depth = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    outputs = rasterizer(
        means3D = means3D_final, means2D = means2D, shs = shs_final,
        colors_precomp = colors_precomp, opacities = opacity_final, 
        scales = scales_final, rotations = rotations_final, cov3D_precomp = cov3D_precomp
    )

    if len(outputs) == 2:
        rendered_image, radii = outputs
    elif len(outputs) == 3:
        rendered_image, radii, depth = outputs

    trbfdistanceoffset = cam.emb_timestamp*pointtimes
    spacetime_mask_a = pc.get_motion_mask(trbfdistanceoffset.detach())

    rendered_weight, _, _ = w_rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = spacetime_mask_a,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)


    return dict(
        image=rendered_image,
        radii=radii,
        depth=depth,
        weight = rendered_weight
    )

def render_4dgs(pc, cam, screenspace_points, rasterizer, w_rasterizer, pointtimes, pipe_2):
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation
    scales_t = pc.get_scaling_t
    ts = pc.get_t
    rotations_r = pc.get_rotation_r

    shs = None
    colors_precomp = None
    shs = pc.get_features
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])

    means3D, means2D, shs, flow_2d, ts, scales, scales_t, rotations, rotations_r, opacity = \
            freeze_tensors(means3D, means2D, shs, flow_2d, ts, scales, scales_t, rotations, rotations_r, opacity)

    cov3D_precomp = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, flow, covs_com = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp)
    

    if pipe_2.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = cam.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]

    trbfdistanceoffset = cam.emb_timestamp*pointtimes
    spacetime_mask = pc.get_motion_mask(trbfdistanceoffset.detach())

    rendered_weight, _, _ = w_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = spacetime_mask,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return dict(
        image=rendered_image,
        radii=radii,
        depth=depth,
        flow=flow,
        alpha=alpha,
        covs_com=covs_com,
        weight = rendered_weight
    )


def render_E3(
    viewpoint_camera, pc, pc_e, pc_h, 
    pipe, bg_color : torch.Tensor, timestamp=None, scaling_modifier=1.0, 
    override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0,
    iter=None, num_down_emb_c=5, num_down_emb_f=5, cam_no=None, basicfunction = None):
    """
    Unified 3-expert rendering function.

    Combines four experts — Ex4DGS, E-D3DGS, and 4DGaussians —
    and renders them within a unified pipeline for MoE-GS.

    Rasterizer configuration:
    • Ex4DGS  →  GaussianRasterizer (shared with STG)
    • E-D3DGS →  EGaussianRasterizer (shared with 4DGaussians)
    • 4DGaussians    →  EGaussianRasterizer (same as E-D3DGS)
    • Weight-splatting (per-Gaussian weight)) → LGRzer (shared across all experts)

    Returns:
        dict[str, torch.Tensor]: Rendered images, masks, and auxiliary data
        for each expert and the MoE fusion stage.
    """
    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_e = torch.zeros_like(pc_e.get_xyz, dtype=pc_e.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_h = torch.zeros_like(pc_h.get_xyz, dtype=pc_h.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    pointtimes = torch.ones((pc.get_xyz_at_t(timestamp, mode=1).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_motion = torch.ones((pc.get_xyz_at_t(timestamp, mode=2).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_e = torch.ones((pc_e.get_xyz.shape[0],1), dtype=pc_e.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_h = torch.ones((pc_h.get_xyz.shape[0],1), dtype=pc_h.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    e_raster_settings = EGaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )
    e_rasterizer = EGaussianRasterizer(raster_settings=e_raster_settings)

    """ SpaceTime Gaussian attribute """
    s_raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    s_rasterizer = GaussianRasterizer(raster_settings=s_raster_settings)
    
    m_raster_settings = LGRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)
    w_rasterizer = LGRzer(raster_settings=m_raster_settings)

    output = render_ex4dgs(pc, viewpoint_camera, screenspace_points, rasterizer, w_rasterizer, timestamp, pointtimes, pointtimes_motion, mode, training)
    e_output = render_ed3dgs(pc_e, viewpoint_camera, screenspace_points_e, e_rasterizer, w_rasterizer, pointtimes_e, cam_no, iter, num_down_emb_c, num_down_emb_f)
    h_output = render_4dgaussians(pc_h, viewpoint_camera, screenspace_points_h, e_rasterizer, w_rasterizer, pointtimes_h)
    
    # time-aware weighting
    stacked_weights = torch.stack([output['weight'], e_output['weight'], h_output['weight']], dim=0)  # (5, C, H, W)
    temp_weights = pc.rgbdecoder(stacked_weights, viewpoint_camera.rays)  # (5, 6, H, W)
    output['weight'], e_output['weight'], h_output['weight'] = temp_weights.unbind(dim=0)

    torch.cuda.synchronize()

    return {
            "render": output['image'],
            "e_render": e_output['image'],
            "h_render" : h_output['image'],
            "weight": output['weight'].squeeze(0),
            "e_weight": e_output['weight'].squeeze(0),
            "h_weight" : h_output['weight'].squeeze(0),
            "depth": output['depth'],
            "opticalflow": output['out_flow'],
            "acc": output['acc'],
            "viewspace_points": output['screenspace_points'],
            "viewspace_l1points": output['flow'],
            "dominent_idxs": output['idxs'],
            "visibility_filter" : output['radii'] > 0,
            "radii": output['radii']
            }


def render_E3_tech(
    viewpoint_camera, pc, pc_e, pc_s_fh, 
    pipe, bg_color : torch.Tensor, b_bg_color : torch.Tensor, timestamp=None, scaling_modifier=1.0, 
    override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0,
    iter=None, num_down_emb_c=5, num_down_emb_f=5, cam_no=None, basicfunction = None):
    """
    Unified 3-expert rendering function (Technicolor configuration).

    This function is designed specifically for the Technicolor Dataset,
    which contains approximately 50 frames per sequence. 
    In this short-duration setup, only a single STG model is used 
    (unlike N3V, which requires two STG models for 0-149 and 150-299 frames).

    Rasterizer configuration:
    • Ex4DGS  →  GaussianRasterizer (shared with STG)
    • E-D3DGS →  EGaussianRasterizer (shared with 4DGaussians)
    • STG     →  GaussianRasterizer (same as Ex4DGS)
    • Weight-splatting (per-Gaussian weight)) → LGRzer (shared across all experts)

    Returns:
        dict[str, torch.Tensor]: Rendered images, masks, and auxiliary data
        for each expert and the MoE fusion stage.
    """
    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_e = torch.zeros_like(pc_e.get_xyz, dtype=pc_e.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_s_fh = torch.zeros_like(pc_s_fh.get_xyz, dtype=pc_s_fh.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    pointtimes = torch.ones((pc.get_xyz_at_t(timestamp, mode=1).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_motion = torch.ones((pc.get_xyz_at_t(timestamp, mode=2).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_e = torch.ones((pc_e.get_xyz.shape[0],1), dtype=pc_e.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    e_raster_settings = EGaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )
    e_rasterizer = EGaussianRasterizer(raster_settings=e_raster_settings)

    s_raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=b_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    s_rasterizer = GaussianRasterizer(raster_settings=s_raster_settings)

    m_raster_settings = LGRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)
    w_rasterizer = LGRzer(raster_settings=m_raster_settings)

    output = render_ex4dgs(pc, viewpoint_camera, screenspace_points, rasterizer, w_rasterizer, timestamp, pointtimes, pointtimes_motion, mode, training)
    e_output = render_ed3dgs(pc_e, viewpoint_camera, screenspace_points_e, e_rasterizer, w_rasterizer, pointtimes_e, cam_no, iter, num_down_emb_c, num_down_emb_f)

    pointtimes_s = torch.ones((pc_s_fh.get_xyz.shape[0],1), dtype=pc_s_fh.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
    fh_timestamp = viewpoint_camera.emb_timestamp
    s_output = render_stg(pc_s_fh, viewpoint_camera, screenspace_points_s_fh, pointtimes_s, s_rasterizer, w_rasterizer, fh_timestamp, basicfunction)
    
    # time-aware weighting
    stacked_weights = torch.stack([output['weight'], e_output['weight'], s_output['weight']], dim=0)  # (5, C, H, W)
    temp_weights = pc.rgbdecoder(stacked_weights, viewpoint_camera.rays)  # (5, 6, H, W)
    output['weight'], e_output['weight'], s_output['weight'] = temp_weights.unbind(dim=0)

    torch.cuda.synchronize()

    return {
            "render": output['image'],
            "e_render": e_output['image'],
            "s_render" : s_output['image'],
            "weight": output['weight'].squeeze(0),
            "e_weight": e_output['weight'].squeeze(0),
            "s_weight" : s_output['weight'].squeeze(0),
            "depth": output['depth'],
            "opticalflow": output['out_flow'],
            "acc": output['acc'],
            "viewspace_points": output['screenspace_points'],
            "viewspace_l1points": output['flow'],
            "dominent_idxs": output['idxs'],
            "visibility_filter" : output['radii'] > 0,
            "radii": output['radii']
            }


def render_E4(
    viewpoint_camera, pc, pc_e, pc_s_fh, pc_s_sh, pc_h, 
    pipe, bg_color : torch.Tensor, fd_bg_color : torch.Tensor, timestamp=None, scaling_modifier=1.0, 
    override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0,
    iter=None, num_down_emb_c=5, num_down_emb_f=5, cam_no=None, basicfunction = None):
    """
    Unified 4-expert rendering function.

    Combines four experts — Ex4DGS, E-D3DGS, STG, and 4DGaussians —
    and renders them within a unified pipeline for MoE-GS.

    Rasterizer configuration:
    • Ex4DGS  →  GaussianRasterizer (shared with STG)
    • E-D3DGS →  EGaussianRasterizer (shared with 4DGaussians)
    • STG     →  GaussianRasterizer (same as Ex4DGS)
    • 4DGaussians    →  EGaussianRasterizer (same as E-D3DGS)
    • Weight-splatting (per-Gaussian weight)) → LGRzer (shared across all experts)

    Returns:
        dict[str, torch.Tensor]: Rendered images, masks, and auxiliary data
        for each expert and the MoE fusion stage.
    """
    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_e = torch.zeros_like(pc_e.get_xyz, dtype=pc_e.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_s_fh = torch.zeros_like(pc_s_fh.get_xyz, dtype=pc_s_fh.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_s_sh = torch.zeros_like(pc_s_sh.get_xyz, dtype=pc_s_sh.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_h = torch.zeros_like(pc_h.get_xyz, dtype=pc_h.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    pointtimes = torch.ones((pc.get_xyz_at_t(timestamp, mode=1).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_motion = torch.ones((pc.get_xyz_at_t(timestamp, mode=2).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_e = torch.ones((pc_e.get_xyz.shape[0],1), dtype=pc_e.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_h = torch.ones((pc_h.get_xyz.shape[0],1), dtype=pc_h.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    e_raster_settings = EGaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )
    e_rasterizer = EGaussianRasterizer(raster_settings=e_raster_settings)

    """ SpaceTime Gaussian attribute """
    s_raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=fd_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    s_rasterizer = GaussianRasterizer(raster_settings=s_raster_settings)
    
    m_raster_settings = LGRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)
    w_rasterizer = LGRzer(raster_settings=m_raster_settings)

    output = render_ex4dgs(pc, viewpoint_camera, screenspace_points, rasterizer, w_rasterizer, timestamp, pointtimes, pointtimes_motion, mode, training)
    e_output = render_ed3dgs(pc_e, viewpoint_camera, screenspace_points_e, e_rasterizer, w_rasterizer, pointtimes_e, cam_no, iter, num_down_emb_c, num_down_emb_f)
    h_output = render_4dgaussians(pc_h, viewpoint_camera, screenspace_points_h, e_rasterizer, w_rasterizer, pointtimes_h)

    if timestamp <150:
        pointtimes_s = torch.ones((pc_s_fh.get_xyz.shape[0],1), dtype=pc_s_fh.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
        fh_timestamp = (timestamp/150)
        s_output = render_stg(pc_s_fh, viewpoint_camera, screenspace_points_s_fh, pointtimes_s, s_rasterizer, w_rasterizer, fh_timestamp, basicfunction)
    else:
        pointtimes_s = torch.ones((pc_s_sh.get_xyz.shape[0],1), dtype=pc_s_sh.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
        sh_timestamp = ((timestamp-150)/150)
        s_output = render_stg(pc_s_sh, viewpoint_camera, screenspace_points_s_sh, pointtimes_s, s_rasterizer, w_rasterizer, sh_timestamp, basicfunction)
    
    # time-aware weighting
    stacked_weights = torch.stack([output['weight'], e_output['weight'], s_output['weight'], h_output['weight']], dim=0)  # (5, C, H, W)
    temp_weights = pc.rgbdecoder(stacked_weights, viewpoint_camera.rays)  # (5, 6, H, W)
    output['weight'], e_output['weight'], s_output['weight'], h_output['weight'] = temp_weights.unbind(dim=0)

    torch.cuda.synchronize()

    return {
            "render": output['image'],
            "e_render": e_output['image'],
            "s_render" : s_output['image'],
            "h_render" : h_output['image'],
            "weight": output['weight'].squeeze(0),
            "e_weight": e_output['weight'].squeeze(0),
            "s_weight" : s_output['weight'].squeeze(0),
            "h_weight" : h_output['weight'].squeeze(0),
            "depth": output['depth'],
            "opticalflow": output['out_flow'],
            "acc": output['acc'],
            "viewspace_points": output['screenspace_points'],
            "viewspace_l1points": output['flow'],
            "dominent_idxs": output['idxs'],
            "visibility_filter" : output['radii'] > 0,
            "radii": output['radii']
            }

def render_E5(
    viewpoint_camera, pc, pc_e, pc_f, pc_s_fh, pc_s_sh, pc_h, 
    pipe, pipe_2, bg_color : torch.Tensor, fd_bg_color :torch.Tensor, timestamp=None, scaling_modifier=1.0, 
    override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0,
    iter=None, num_down_emb_c=5, num_down_emb_f=5, cam_no=None, basicfunction = None):
    """
    Unified 5-expert rendering function.

    Combines five experts — Ex4DGS, E-D3DGS, 4DGS, STG, and 4DGaussians —
    and renders them within a unified pipeline for MoE-GS.

    Rasterizer configuration:
    • Ex4DGS  →  GaussianRasterizer (shared with STG)
    • E-D3DGS →  EGaussianRasterizer (shared with 4DGaussians)
    • 4DGS    →  FGaussianRasterizer
    • STG     →  GaussianRasterizer (same as Ex4DGS)
    • 4DGaussians    →  EGaussianRasterizer (same as E-D3DGS)
    • Weight-splatting (per-Gaussian weight)) → LGRzer (shared across all experts)

    Returns:
        dict[str, torch.Tensor]: Rendered images, masks, and auxiliary data
        for each expert and the MoE fusion stage.
    """
    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_e = torch.zeros_like(pc_e.get_xyz, dtype=pc_e.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_f = torch.zeros_like(pc_f.get_xyz, dtype=pc_f.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_s_fh = torch.zeros_like(pc_s_fh.get_xyz, dtype=pc_s_fh.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_s_sh = torch.zeros_like(pc_s_sh.get_xyz, dtype=pc_s_sh.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_h = torch.zeros_like(pc_h.get_xyz, dtype=pc_h.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    pointtimes = torch.ones((pc.get_xyz_at_t(timestamp, mode=1).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_motion = torch.ones((pc.get_xyz_at_t(timestamp, mode=2).shape[0],1), dtype=pc._xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_e = torch.ones((pc_e.get_xyz.shape[0],1), dtype=pc_e.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    pointtimes_f = torch.ones((pc_f.get_xyz.shape[0],1), dtype=pc_f.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
    pointtimes_h = torch.ones((pc_h.get_xyz.shape[0],1), dtype=pc_h.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    e_raster_settings = EGaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )
    e_rasterizer = EGaussianRasterizer(raster_settings=e_raster_settings)

    fd_tanfovx = math.tan(viewpoint_camera.fd_FoVx * 0.5)
    fd_tanfovy = math.tan(viewpoint_camera.fd_FoVy * 0.5)
    f_raster_settings = FGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=fd_tanfovx,
        tanfovy=fd_tanfovy,
        bg=fd_bg_color if not pipe_2.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.fd_world_view_transform,
        projmatrix=viewpoint_camera.fd_full_proj_transform,
        sh_degree=pc_f.active_sh_degree,
        sh_degree_t=pc_f.active_sh_degree_t,
        campos=viewpoint_camera.fd_camera_center,
        timestamp= viewpoint_camera.fd_timestamp,
        time_duration=pc_f.time_duration[1]-pc_f.time_duration[0],
        rot_4d=pc_f.rot_4d,
        gaussian_dim=pc_f.gaussian_dim,
        force_sh_3d=pc_f.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )
    f_rasterizer = FGaussianRasterizer(raster_settings=f_raster_settings)

    """ SpaceTime Gaussian attribute """
    s_raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=fd_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )
    s_rasterizer = GaussianRasterizer(raster_settings=s_raster_settings)
    
    m_raster_settings = LGRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)
    w_rasterizer = LGRzer(raster_settings=m_raster_settings)

    m_4d_raster_settings = LGRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=fd_tanfovx,
        tanfovy=fd_tanfovy,
        # bg=fd_bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.fd_world_view_transform,
        projmatrix=viewpoint_camera.fd_full_proj_transform,
        sh_degree=pc_f.active_sh_degree,
        campos=viewpoint_camera.fd_camera_center,
        prefiltered=False)
    w_rasterizer_f = LGRzer(raster_settings=m_4d_raster_settings)

    output = render_ex4dgs(pc, viewpoint_camera, screenspace_points, rasterizer, w_rasterizer, timestamp, pointtimes, pointtimes_motion, mode, training)
    f_output = render_4dgs(pc_f, viewpoint_camera, screenspace_points_f, f_rasterizer, w_rasterizer_f, pointtimes_f, pipe_2)
    e_output = render_ed3dgs(pc_e, viewpoint_camera, screenspace_points_e, e_rasterizer, w_rasterizer, pointtimes_e, cam_no, iter, num_down_emb_c, num_down_emb_f)
    h_output = render_4dgaussians(pc_h, viewpoint_camera, screenspace_points_h, e_rasterizer, w_rasterizer, pointtimes_h)

    if timestamp <150:
        pointtimes_s = torch.ones((pc_s_fh.get_xyz.shape[0],1), dtype=pc_s_fh.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
        fh_timestamp = (timestamp/150)
        s_output = render_stg(pc_s_fh, viewpoint_camera, screenspace_points_s_fh, pointtimes_s, s_rasterizer, w_rasterizer, fh_timestamp, basicfunction)
    else:
        pointtimes_s = torch.ones((pc_s_sh.get_xyz.shape[0],1), dtype=pc_s_sh.get_xyz.dtype, requires_grad=False, device="cuda") + 0 #
        sh_timestamp = ((timestamp-150)/150)
        s_output = render_stg(pc_s_sh, viewpoint_camera, screenspace_points_s_sh, pointtimes_s, s_rasterizer, w_rasterizer, sh_timestamp, basicfunction)

    # time-aware weighting
    stacked_weights = torch.stack([output['weight'], e_output['weight'], f_output['weight'], s_output['weight'], h_output['weight']], dim=0)  # (5, C, H, W)
    temp_weights = pc.rgbdecoder(stacked_weights, viewpoint_camera.rays)  # (5, 6, H, W)
    output['weight'], e_output['weight'], f_output['weight'], s_output['weight'], h_output['weight'] = temp_weights.unbind(dim=0)

    torch.cuda.synchronize()

    return {
            "render": output['image'],
            "e_render": e_output['image'],
            "f_render": f_output['image'],
            "s_render" : s_output['image'],
            "h_render" : h_output['image'],
            "weight": output['weight'].squeeze(0),
            "e_weight": e_output['weight'].squeeze(0),
            "f_weight": f_output['weight'].squeeze(0),
            "s_weight" : s_output['weight'].squeeze(0),
            "h_weight" : h_output['weight'].squeeze(0),
            "depth": output['depth'],
            "opticalflow": output['out_flow'],
            "acc": output['acc'],
            "viewspace_points": output['screenspace_points'],
            "viewspace_l1points": output['flow'],
            "dominent_idxs": output['idxs'],
            "visibility_filter" : output['radii'] > 0,
            "radii": output['radii']
            }

