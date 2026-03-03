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
import random
import json
from utils.system_utils import searchForMaxIteration
from thirdparty.Hexplane.scene.gaussian_model import HGaussianModel
from thirdparty.Hexplane.arguments import ModelParams

class HScene:

    gaussians : HGaussianModel

    def __init__(self, args : ModelParams, gaussians : HGaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False,  render=False, save_path=None, cameras_extent=0.0, init_mixprob=0.9):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.fgaussian_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.cameras_extent = cameras_extent

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        gaussians.duration = args.duration

        if render:
            self.model_path = save_path
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "4DGaussians",
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"), self.cameras_extent, init_mixprob)
            self.gaussians.load_model(os.path.join(self.model_path,
                                                "4DGaussians",
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                ))
        else:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"), self.cameras_extent, init_mixprob, search=True)
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter)
                                                ))


    def save(self, iteration, save_path):
        if save_path != None:
            self.model_path = save_path
        point_cloud_path = os.path.join(self.model_path, "4DGaussians/point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)