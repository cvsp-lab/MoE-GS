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
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from thirdparty.reparmetrize.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from thirdparty.reparmetrize.utils.data_utils import CameraDataset

class FScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.fourd_ckpt_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

    def save(self, iteration, save_path):
        self.model_path = save_path
        os.makedirs(os.path.join(self.model_path, '4DGS'), exist_ok=True)
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/4DGS" + "/chkpnt" + str(iteration) + ".pth")