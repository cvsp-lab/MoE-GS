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
from thirdparty.polynomial.scene.dataset_readers import sceneLoadTypeCallbacks
from thirdparty.polynomial.scene.oursfull import SGaussianModel
from arguments import ModelParams
from PIL import Image 
from thirdparty.polynomial.utils.camera_utils import camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosv2nogt
from thirdparty.polynomial.helper_train import recordpointshelper, getfisheyemapper
import torch 
class SScene:

    # gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], multiview=False,duration=150.0, loader="colmap", render=False, save_path=None, cameras_extent=0.0, stg_path=None, init_mixprob=0.9):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = stg_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.refmodelpath = None
        self.cameras_extent = cameras_extent


        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
        
        gaussians.duration = 150

        if render:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud.ply"), self.cameras_extent, init_mixprob)
        else:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"), self.cameras_extent, init_mixprob, search=True)


    def save(self, iteration, save_path):
        if save_path != None:
            self.model_path = save_path
        point_cloud_path = os.path.join(self.model_path, "point_cloud", "iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # recordpointshelper(model_path, numpoints, iteration, string):
    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)