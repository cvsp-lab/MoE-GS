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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.stg_path = ""
        self.emb_path = ""
        self.fourd_ckpt_path = ""
        self.fgaussian_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.model = "cubic"
        self.loader = "neural3dvideo" 
        self.interp_type = "cube" 
        self.rot_interp_type = "slerp" 
        self.lazy_loader = True
        self.llffhold = 8
        self.time_interval = 5
        self.time_pad = 3
        self.var_pad = 3
        self.time_pad_type = 0 # 0: none, 1: reflect 2: repeat
        self.kernel_size = 0.1
        self.start_duration = 5
        self.duration = -1
        self.sample_every = 1
        self.progressive_step = 1
        self.start_timestamp = 0
        self.end_timestamp = -1
        self.near = 0.2
        self.far = 300.0
        self._exp_name = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.eval_shfs_4d = False
        self.env_map_res = 0
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.dynamic_position_lr_init = 0.00016
        self.dynamic_position_lr_final = 0.000016
        self.dynamic_position_lr_delay_mult = 0.01
        self.dynamic_position_lr_max_steps = 30_000
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.00001
        self.disp_lr = 0.0001
        self.feature_motion_lr = 0.0025
        self.rotation_motion_lr = 0.001
        self.opacity_motion_lr = 0.05
        self.opacity_motion_center_lr = 0.001
        self.opacity_motion_var_lr = 0.0005
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.l1_accum = True
        self.prune_threshold = 0.00008
        self.prune_STG_threshold= 0.00008
        self.prune_deform_threshold = 0.00008
        self.prune_A_threshold = 0.00008
        self.prune_weight_interval = 500
        self.prune_from_iter = 500
        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.extract_from_iter = 500
        self.densify_until_iter = 15_000
        self.progressive_growing_steps = 300
        self.error_base_prune_steps = 20000
        self.ssim_prune_every = 5
        self.l1_prune_every = 5
        self.make_dynamic_interval = 200 # != densification_interval
        self.extracton_interval = 3000
        self.extract_every = 1
        self.extract_percentile = 0.98
        self.prune_invisible_interval = 6000
        self.densify_grad_threshold = 0.0002
        self.densify_dgrad_threshold = 0.0001
        self.s_max_ssim = 0.6
        self.s_l1_thres = 0.08
        self.d_max_ssim = 0.6
        self.d_l1_thres = 0.08
        self.static_reg = 0.0001
        self.motion_reg = 0.0001
        self.rot_reg = 0.00
        self.coord_reg = 0.00
        self.random_background = True
        self.position_t_lr_init = -1.0

        # Ex4DGS additional attributes
        self.feature_lr = 0.0025
        self.featuret_lr = 0.001
        self.rgb_lr = 0.0001

        # E-D3DGS deformation lr
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.deformation_lr_max_steps = 60_000
        self.offsets_lr = 0.00002
        self.feature_lr_div_factor = 20.0

        # spacetime addtional parameters
        self.omega_lr = 0.0001
        self.trbfc_lr = 0.0001 # 
        self.trbfs_lr = 0.03
        self.movelr = 3.5
        self.trbfslinit = 0.0 # 
        self.preprocesspoints = 0  
        self.addsphpointsscale = 0.8  
        self.raystart = 0.7
        # self.rgb_lr = 0.0001

        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        super().__init__(parser, "Optimization Parameters")

class ModelEmbParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.defor_depth = 1
        self.min_embeddings = 30
        self.max_embeddings = 150
        self.no_ds=False
        self.no_dr=False
        self.no_do=True
        self.no_dc=False
        
        self.temporal_embedding_dim=256
        self.gaussian_embedding_dim=32
        self.use_coarse_temporal_embedding=False
        self.no_c2f_temporal_embedding=False
        self.no_coarse_deform=False
        self.no_fine_deform=False
        
        self.total_num_frames=300
        self.c2f_temporal_iter=20000
        self.deform_from_iter=0
        self.use_anneal=True
        self.zero_temporal=False
        super().__init__(parser, "ModelHiddenParams")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width_2 = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth_2 = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds_2=False # cancel the deformation of Gaussians' scaling
        self.no_dr_2=False # cancel the deformation of Gaussians' rotations
        self.no_do_2=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        
        super().__init__(parser, "ModelHiddenParams_4DGS")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.save_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
