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

import torch
from torch import nn
import numpy as np
from kornia import create_meshgrid
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV, pix2ndc, ndc2pix, fov2focal
from thirdparty.reparmetrize.utils.graphics_utils import getProjectionMatrixCenterShift
from utils.general_utils import PILtoTorch
from PIL import Image
import os

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Cameravideo():
    def __init__(self, colmap_id, R, T, FoVx, FoVy, gt_alpha_mask, image,
                 image_name, img_name, image_path, cam_image_dir, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 near=0.01, far=100.0, timestamp=0.0, emb_timestamp=0.0, fd_timestamp=0.0,
                 rayo=None, rayd=None, rays=None, cxr=0.0, cyr=0.0, resolution=(1., 1.), cam_no=0,
                 fd_FovX=None, fd_FovY=None, fd_cx=None, fd_cy=None, fd_fl_x=None, fd_fl_y=None, fd_R=None, fd_T=None,
                 opticalflow_path=None, depth_path=None, im_scale=1.0
                 ):
        super(Cameravideo, self).__init__()
        
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.img_name = img_name 
        self.cam_image_dir = cam_image_dir
        self.timestamp = timestamp
        self.emb_timestamp = emb_timestamp
        self.fd_timestamp = fd_timestamp
        self.fisheyemapper = None
        self.resolution = resolution
        self.image_path = image_path
        self.opticalflow_path = opticalflow_path
        self.depth_path = depth_path
        self.im_scale = im_scale
        self.cam_no = cam_no
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = far
        self.znear = near
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
        
        self.fd_R = fd_R
        self.fd_T = fd_T
        self.fd_FoVx = fd_FovX
        self.fd_FoVy = fd_FovY
        self.fd_cx = fd_cx
        self.fd_cy = fd_cy
        self.fd_fl_x = fd_fl_x
        self.fd_fl_y = fd_fl_y

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.fd_world_view_transform = torch.tensor(getWorld2View2(fd_R, fd_T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        if fd_cx > 0:
            self.fd_projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, fd_cx, fd_cy, fd_fl_x, fd_fl_y, self.image_width, self.image_height).transpose(0,1).cuda()
        else:
            self.fd_projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fd_FoVx, fovY=self.fd_FoVy).transpose(0,1).cuda()
        self.fd_full_proj_transform = (self.fd_world_view_transform.unsqueeze(0).bmm(self.fd_projection_matrix.unsqueeze(0))).squeeze(0)
        self.fd_camera_center = self.fd_world_view_transform.inverse()[3, :3]

        self.image = image
        
    def get_rays(self):
        grid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False)[0] + 0.5
        i, j = grid.unbind(-1)
        pts_view = torch.stack([(i-self.fd_cx)/self.fd_fl_x, (j-self.fd_cy)/self.fd_fl_y, torch.ones_like(i), torch.ones_like(i)], -1).to(self.data_device)
        c2w = torch.linalg.inv(self.fd_world_view_transform.transpose(0, 1))
        pts_world =  pts_view @ c2w.T
        directions = pts_world[...,:3] - self.fd_camera_center[None,None,:]
        return self.fd_camera_center[None,None], directions / torch.norm(directions, dim=-1, keepdim=True)
WARNED = False



class Cameravideo2():
    def __init__(self, colmap_id, R, T, D_T, FoVx, FoVy, gt_alpha_mask, image,
                 image_name, cam_name, image_path, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 near=0.01, far=100.0, timestamp=0.0, emb_timestamp=0.0,
                 rayo=None, rayd=None, rays=None, cxr=0.0, cyr=0.0, resolution=(1., 1.), cam_no=0,
                 opticalflow_path=None, depth_path=None, im_scale=1.0
                 ):
        super(Cameravideo2, self).__init__()
        
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.R = R
        self.T = T
        self.D_T = D_T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.emb_timestamp = emb_timestamp
        self.fisheyemapper = None
        self.resolution = resolution
        self.image_path = image_path
        self.opticalflow_path = opticalflow_path
        self.depth_path = depth_path
        self.im_scale = im_scale
        self.cam_no = int(cam_no)
        self.cam_name = cam_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = far
        self.znear = near
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        self.world_view_transform_2 = torch.tensor(getWorld2View2(R, D_T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()

        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.full_proj_transform_2 = (self.world_view_transform_2.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_center_2 = self.world_view_transform_2.inverse()[3, :3]

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
        
        self.zfar = 100.0
        self.znear = 0.01
        self.image = image

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    

def loadCamVideo(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 

    cam_no = int(os.path.dirname(cam_info.image_path).split('/')[-1][3:])
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None
    
    """4D Gaussian Splatting camera param"""
    fd_cx = cam_info.fd_cx / scale
    fd_cy = cam_info.fd_cy / scale
    fd_fl_y = cam_info.fd_fl_y / scale
    fd_fl_x = cam_info.fd_fl_x / scale

    return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, 
                  image=cam_info.image,
                  image_name=cam_info.image_name, img_name=cam_info.img_name, image_path=cam_info.image_path, cam_image_dir=cam_info.cam_image_dir, uid=id, data_device=args.data_device, 
                  near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, emb_timestamp=cam_info.emb_timestamp, fd_timestamp=cam_info.fd_timestamp,
                  rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution, cam_no=cam_no, \
                  fd_FovX=cam_info.fd_FovX, fd_FovY=cam_info.fd_FovY, fd_cx=fd_cx, fd_cy=fd_cy, fd_fl_x=fd_fl_x, fd_fl_y=fd_fl_y, fd_R=cam_info.fd_R, fd_T=cam_info.fd_T,)


def loadCamVideo3(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 

    cam_no = cam_info.image_name.split('_')[-1][:2]
    
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None
    

    return Cameravideo2(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, D_T=cam_info.D_T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, 
                  image=cam_info.image,
                  image_name=cam_info.image_name, cam_name=cam_info.cam_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                  near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, emb_timestamp=cam_info.emb_timestamp,
                  rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution, cam_no=cam_no)    

def loadCamVideoss(args, id, cam_info, resolution_scale, nogt=False):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

    resolution = (int(orig_w / 2), int(orig_h / 2))
    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 

    im_scale = 1
    # load gt image 
    if nogt == False :
        if "01_Welder" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
                
        if "12_Cave" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
        
        if "04_Truck" in args.source_path:
            if "camera_0008" in cam_info.image_name:
                im_scale = 1.2
        
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution, im_scale=im_scale)
    else:
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution)
        

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCamVideo(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo2(cam_infos, resolution_scale, args, ss=False):
    camera_list = []

    if not ss: #
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideo(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideoss(args, id, c, resolution_scale))

    return camera_list

def cameraList_from_camInfosVideo3(cam_infos, resolution_scale, args, ss=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCamVideo3(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
