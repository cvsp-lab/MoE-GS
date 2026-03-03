# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch 
# from thirdparty.graphics_utils import BasicPointCloud
import numpy as np
from simple_knn._C import distCUDA2
# from mmcv.ops import knn
import torch.nn as nn



# class Sandwich(nn.Module):
#     def __init__(self, dim, outdim=3, bias=False):
#         super(Sandwich, self).__init__()
        
#         self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 

#         self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
#         self.relu = nn.ReLU()

#         self.sigmoid = torch.nn.Sigmoid()


#     def forward(self, input, rays, time=None):
#         albedo, spec, timefeature = input.chunk(3,dim=1)
#         specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
#         specular = self.mlp1(specular)
#         specular = self.relu(specular)
#         specular = self.mlp2(specular)

#         result = albedo + specular
#         result = self.sigmoid(result) 
#         return result


class Sandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec, tim = input.chunk(3,dim=1)
        specular = torch.cat([spec, tim, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular

        return result

class SandwichTimembed(nn.Module):
    def __init__(self, dim=6, outdim=3, bias=False):
        super(SandwichTimembed, self).__init__()
        
        self.mlp1 = nn.Conv2d(18, 12, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 
        self.mlp3 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        # self.ray_film = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)  # gamma & beta 생성
        self.time_mlp_conv = nn.Conv2d(1, 6, kernel_size=1, bias=False)  
        # self.time_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)


    def forward(self, input, rays, time=None):
        albedo, spec, tim = input.chunk(3,dim=1)

        B = spec.shape[0]
        time = torch.tensor(time, dtype=torch.float32, device="cuda").view(1, 1, 1, 1).repeat(B, 1, 1, 1)  # (B, 1, 1, 1)
        time_expanded = time.expand(-1, -1, spec.shape[2], spec.shape[3])  # (B, 1, H, W)
        time_embedding = self.time_mlp_conv(time_expanded)  # (B, 4, H, W)
        rays = rays.expand(B, -1, -1, -1)  # (B, C, H, W)  메모리 복사 없이 확장.

        # ray_film_params = self.ray_film(rays)  # (B, 2C_spec, H, W)
        # gamma, beta = ray_film_params.chunk(2, dim=1)  # (B, C_spec, H, W) x 2
        # spec = gamma * spec + beta  # FiLM 적용

        # B, C, H, W = tim.shape
        # tim_flat = tim.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        # time_embedding_flat = time_embedding.view(B, time_embedding.shape[1], -1).permute(0, 2, 1)  # (B, HW, C)

        # tim_attended, _ = self.time_attn(tim_flat, time_embedding_flat, time_embedding_flat)  # Cross Attention
        # tim_attended = tim_attended.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)로 reshape

        # # 기존 tim과 Attention 후 정보를 합치기
        # tim = tim + tim_attended

        specular = torch.cat([spec, tim, rays, time_embedding], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular)

        result = albedo + specular

        return result
    

class Sandwich_wo_time(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich_wo_time, self).__init__()
        
        self.mlp1 = nn.Conv2d(9, 6, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(3, 1, kernel_size=1, bias=bias)

        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec = input.chunk(2,dim=1)

        rays = rays.expand(spec.shape[0], -1, -1, -1)  # (B, C, H, W)  메모리 복사 없이 확장.
        specular = torch.cat([spec, rays], dim=1) # 3+3 + 5

        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular)

        result = albedo[:, 0:1] + specular
        return result

class LiteSandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(LiteSandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(8, 6, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(3, 1, kernel_size=1, bias=bias)

        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        rays = rays.expand(spec.shape[0], -1, -1, -1) # (B, 6, H, W)
        specular = torch.cat([spec, timefeature, rays], dim=1)

        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular) # (B, 1, H, W)

        result = albedo + specular
        return result
    
class PixelSandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(PixelSandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(2, 6, kernel_size=1, bias=bias) # 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(3, 1, kernel_size=1, bias=bias)

        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        # rays = rays.expand(spec.shape[0], -1, -1, -1) # (B, 6, H, W)
        specular = torch.cat([spec, timefeature], dim=1)

        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular) # (B, 1, H, W)

        result = albedo + specular
        return result

class TinySandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(TinySandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(9, 3, kernel_size=1, bias=bias) # 

        # self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec = input.chunk(2,dim=1)
        specular = torch.cat([spec, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        # specular = self.relu(specular)
        # specular = self.mlp2(specular)

        result = albedo + specular

        return result

# class HeavySandwich(nn.Module):
#     def __init__(self, dim, outdim=3, bias=False):
#         super(HeavySandwich, self).__init__()
        
#         self.mlp1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias) # 
#         self.mlp2 = nn.Conv2d(9, 6, kernel_size=1, bias=bias) # 
#         self.mlp3 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
#         self.relu = nn.ReLU()

#         self.sigmoid = torch.nn.Sigmoid()


#     def forward(self, input, rays, time=None):
#         albedo, spec = input.chunk(2,dim=1)
#         specular = torch.cat([spec, rays], dim=1) # 3+3 + 5
#         specular = self.mlp1(specular)
#         specular = self.relu(specular)
#         specular = self.mlp2(specular)
#         specular = self.relu(specular)
#         specular = self.mlp3(specular)

#         result = albedo + specular

#         return result

class HeavySandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(HeavySandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 9, kernel_size=1, bias=bias) # 
        self.mlp1_1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias) # 
        self.mlp1_2 = nn.Conv2d(9, 9, kernel_size=1, bias=bias) # 
        self.mlp1_3 = nn.Conv2d(9, 9, kernel_size=1, bias=bias) # 

        self.mlp2 = nn.Conv2d(9, 6, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec, tim = input.chunk(3,dim=1)
        specular = torch.cat([spec, tim, rays], dim=1) # 3+3 + 5
        
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp1_1(specular)
        specular = self.relu(specular)
        specular = self.mlp1_2(specular)
        specular = self.relu(specular)
        specular = self.mlp1_3(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular)

        result = albedo + specular

        return result
    

    
class LogBlock(nn.Module):
    """LoG Filter Block for LoG loss"""
    def __init__(self):
        super(LogBlock, self).__init__()

        ## RGB to Gray Block
        np_filter1 = np.array([[0.2989, 0.5870, 0.1140]]);
        conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(np_filter1).float().unsqueeze(2).unsqueeze(2))

        ## LoG Filter Block
        np_filter_2 =np.array([[0, -1, 0] ,[-1 ,4 ,-1] ,[0 ,-1 ,0]])
        conv2 =nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight =nn.Parameter(torch.from_numpy(np_filter_2).float().unsqueeze(0).unsqueeze(0))

        self.main = nn.Sequential(conv1, conv2)

        ## Fix all weights
        for param in self.main.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.main(x)
    

class Sandwichnoact(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoact, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result

class Sandwichnoactss(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoactss, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)  
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)


        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        return result
    
    
####### following are also good rgb model but not used in the paper, slower than sandwich, inspired by color shift in hyperreel
# remove sigmoid for immersive dataset
class RGBDecoderVRayShift(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(RGBDecoderVRayShift, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dwconv1(input) + input 
        albeado = self.mlp1(x)
        specualr = torch.cat([x, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)   
        return result 
    


def interpolate_point(pcd, N=4):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])
        else:
            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1   
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * 0.25)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd



def interpolate_pointv3(pcd, N=4,m=0.25):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            M = spatialdistance.shape[0]
            num_take = int(M * m)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd




def interpolate_partuse(pcd, N=4):
    # used in ablation study
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            pass
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd



def padding_point(pcd, N=4):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)
    totallength = len(timestamps)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        

        if timeidx != 0 and timeidx != len(timestamps) - 1:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

             
        else:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3


            xyznnpoints = knn(2, xyzinput, xyzinput, False)


            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * 0.125)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])

            if timeidx == 0:
                newtime.append(oldtime[selectedmask][masksnumpy] - (1 / totallength)) 
            else :
                newtime.append(oldtime[selectedmask][masksnumpy] + (1 / totallength))
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd

def getcolormodel(rgbfuntion):
    if rgbfuntion == "sandwich":
        print("sandwich")
        rgbdecoder = Sandwich(9,3)
        # rgbdecoder = HeavySandwich(9,3)
        # rgbdecoder = TinySandwich(9,3)
    elif rgbfuntion == "sandwichnotime":
        rgbdecoder = Sandwich_wo_time(9,3)
    elif rgbfuntion == "sandwichheavy":
        rgbdecoder = HeavySandwich(9,3)
    elif rgbfuntion == "sandwichlite":
        rgbdecoder = LiteSandwich(9,3)
    elif rgbfuntion == "sandwichpixel":
        print("sandwichpixel")
        rgbdecoder = PixelSandwich(9,3)
    elif rgbfuntion == "sandwichtimembed":
        rgbdecoder = SandwichTimembed(6,3)
    elif rgbfuntion == "sandwichnoactss":
        rgbdecoder = Sandwichnoactss(9,3)
    else :
        return None 
    return rgbdecoder

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5



