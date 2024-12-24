"""
Data augmentation for 3D voxel data.
"""

import random
import math
import torch
import torch.nn.functional as F
from skimage import measure

class ComprehensiveAugmentor:
    def __init__(self,
                 rot_prob=0.3, max_rot_deg=30,
                 scale_prob=0.2, scale_range=(0.9,1.1),
                 noise_prob=0.2, noise_std=0.01,
                 deform_prob=0.2, deform_mag=1,
                 occlude_prob=0.2,
                 surface_prob=0.2):
        self.rot_prob = rot_prob
        self.max_rot_deg = max_rot_deg
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.deform_prob = deform_prob
        self.deform_mag = deform_mag
        self.occlude_prob = occlude_prob
        self.surface_prob = surface_prob

    def __call__(self, voxel):
        v = voxel.clone()
        if random.random() < self.rot_prob:
            deg = random.uniform(-self.max_rot_deg, self.max_rot_deg)
            v = self._rotate_voxel_arbitrary(v, deg)
        if random.random() < self.scale_prob:
            sc = random.uniform(*self.scale_range)
            v = self._scale_voxel(v, sc)
        if random.random() < self.noise_prob:
            noise = torch.randn_like(v)*self.noise_std
            v = torch.clamp(v+noise, 0, 1)
        if random.random() < self.deform_prob:
            v = self._random_deform(v, self.deform_mag)
        if random.random() < self.occlude_prob:
            v = self._random_occlusion(v)
        if random.random() < self.surface_prob:
            v = self._surface_perturb(v)
        return v

    def _rotate_voxel_arbitrary(self, voxel, angle_deg):
        C,D,H,W = voxel.shape
        angle_rad = math.radians(angle_deg)
        cosA, sinA = math.cos(angle_rad), math.sin(angle_rad)
        rot_mat = torch.tensor([[ cosA, -sinA, 0,0],
                              [ sinA,  cosA, 0,0],
                              [ 0,     0,    1,0],
                              [ 0,     0,    0,1]], 
                             dtype=torch.float, device=voxel.device)
        vexp = voxel.unsqueeze(0)
        d_lin = torch.linspace(-1,1,D,device=voxel.device)
        h_lin = torch.linspace(-1,1,H,device=voxel.device)
        w_lin = torch.linspace(-1,1,W,device=voxel.device)
        d_coords,h_coords,w_coords = torch.meshgrid(d_lin,h_lin,w_lin,indexing='ij')
        ones = torch.ones_like(d_coords)
        coords = torch.stack([w_coords,h_coords,d_coords,ones],dim=-1)
        coords = coords.view(-1,4)@rot_mat.T
        coords = coords.view(D,H,W,4)
        new_w = coords[...,0]
        new_h = coords[...,1]
        new_d = coords[...,2]
        sampling_grid = torch.stack([new_h,new_w,new_d],dim=-1)
        sampling_grid = sampling_grid.unsqueeze(0)
        rotated = F.grid_sample(vexp,sampling_grid,align_corners=True,mode='bilinear',padding_mode='zeros')
        return rotated[0]

    def _scale_voxel(self, voxel, scale):
        C,D,H,W = voxel.shape
        nD,nH,nW = int(D*scale), int(H*scale), int(W*scale)
        scaled = F.interpolate(voxel.unsqueeze(0), size=(nD,nH,nW),
                             mode='trilinear', align_corners=False)
        scaled = scaled[0]
        out = torch.zeros_like(voxel)
        dmin=(D-nD)//2; dmax=dmin+nD
        hmin=(H-nH)//2; hmax=hmin+nH
        wmin=(W-nW)//2; wmax=wmin+nW
        out[:, dmin:dmax, hmin:hmax, wmin:wmax] = scaled
        return out

    def _random_deform(self, voxel, mag=1):
        C,D,H,W = voxel.shape
        vout = voxel.clone()
        slice_idx = random.randint(0,D-1)
        shift = random.randint(-mag,mag)
        vout[:, slice_idx, :, :] = torch.roll(vout[:, slice_idx, :, :], shifts=shift, dims=-1)
        return vout

    def _random_occlusion(self, voxel):
        C,D,H,W = voxel.shape
        d0 = random.randint(0,D//2)
        d1 = d0+random.randint(1,D//4)
        h0 = random.randint(0,H//2)
        h1 = h0+random.randint(1,H//4)
        w0 = random.randint(0,W//2)
        w1 = w0+random.randint(1,W//4)
        voxel[:, d0:d1,h0:h1,w0:w1] = 0
        return voxel

    def _surface_perturb(self, voxel):
        data_np = (voxel[0].cpu().numpy()>0.5)
        op = random.choice(["dilate","erode"])
        if op=="dilate":
            data_np = measure.binary_dilation(data_np)
        else:
            data_np = measure.binary_erosion(data_np)
        out = torch.from_numpy(data_np.astype(np.float32)).to(voxel.device).unsqueeze(0)
        return out
