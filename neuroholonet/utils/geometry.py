"""
Geometric analysis utilities for 3D voxel data.
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_voxel_laplacian(voxels):
    B,C,D,H,W = voxels.shape
    lap = torch.zeros_like(voxels)
    def val_or_0(b,c,d_,h_,w_):
        if d_<0 or d_>=D or h_<0 or h_>=H or w_<0 or w_>=W:
            return 0.0
        else:
            return voxels[b,c,d_,h_,w_]
    for b in range(B):
        for d_ in range(D):
            for h_ in range(H):
                for w_ in range(W):
                    c0 = val_or_0(b,0,d_,h_,w_)
                    xp = val_or_0(b,0,d_,h_,w_+1)
                    xm = val_or_0(b,0,d_,h_,w_-1)
                    yp = val_or_0(b,0,d_,h_+1,w_)
                    ym = val_or_0(b,0,d_,h_-1,w_)
                    zp = val_or_0(b,0,d_+1,h_,w_)
                    zm = val_or_0(b,0,d_-1,h_,w_)
                    lap[b,0,d_,h_,w_] = xp+xm+yp+ym+zp+zm-6*c0
    return lap

def compute_curvature(voxels):
    lap = compute_voxel_laplacian(voxels)
    return lap.abs()

def connected_components_3d(volume):
    vol_np = volume.cpu().numpy()
    D,H,W = vol_np.shape
    visited = np.zeros_like(vol_np, dtype=bool)
    comps = 0
    def neighbors(d,h,w):
        for dd,hh,ww in [(d-1,h,w),(d+1,h,w),(d,h-1,w),(d,h+1,w),(d,h,w-1),(d,h,w+1)]:
            if 0<=dd<D and 0<=hh<H and 0<=ww<W:
                yield dd,hh,ww
    for d_ in range(D):
        for h_ in range(H):
            for w_ in range(W):
                if vol_np[d_,h_,w_]>0 and not visited[d_,h_,w_]:
                    comps += 1
                    stack = [(d_,h_,w_)]
                    visited[d_,h_,w_] = True
                    while stack:
                        dd,hh,ww = stack.pop()
                        for nd,nh,nw in neighbors(dd,hh,ww):
                            if vol_np[nd,nh,nw]>0 and not visited[nd,nh,nw]:
                                visited[nd,nh,nw] = True
                                stack.append((nd,nh,nw))
    return comps

def compute_topology_measure(voxels):
    B = voxels.shape[0]
    measures = []
    bin_vol = (voxels>0.5).float()
    for b in range(B):
        cc = connected_components_3d(bin_vol[b,0])
        measures.append(cc)
    return torch.tensor(measures, dtype=torch.float32, device=voxels.device)

def sample_voxel_points(vox, n=2048):
    bin_vol = (vox>0.5)
    coords = bin_vol.nonzero(as_tuple=False).float()
    if coords.shape[0]<1:
        return torch.zeros((n,3), device=vox.device)
    if coords.shape[0]>=n:
        idx = torch.randperm(coords.shape[0], device=vox.device)[:n]
        coords = coords[idx]
    else:
        diff = n-coords.shape[0]
        rep = coords[torch.randint(0,coords.shape[0],(diff,), device=vox.device)]
        coords = torch.cat([coords,rep], dim=0)
    return coords

def compute_hausdorff(v1, v2):
    p1 = sample_voxel_points(v1)
    p2 = sample_voxel_points(v2)
    diff = (p1.unsqueeze(1)-p2.unsqueeze(0))
    dist_sq = (diff**2).sum(-1)
    d12 = torch.sqrt(dist_sq.min(dim=1)[0]).max()
    d21 = torch.sqrt(dist_sq.min(dim=0)[0]).max()
    return torch.max(d12,d21).item()

def compute_chamfer(v1, v2):
    p1 = sample_voxel_points(v1)
    p2 = sample_voxel_points(v2)
    diff = (p1.unsqueeze(1)-p2.unsqueeze(0))
    dist_sq = (diff**2).sum(-1)
    d12 = dist_sq.min(dim=1)[0].mean()
    d21 = dist_sq.min(dim=0)[0].mean()
    return (d12+d21).sqrt().item()

def compute_normal_consistency(v1, v2):
    v1 = v1.unsqueeze(0)
    v2 = v2.unsqueeze(0)
    def gradient_3d(v):
        gx = v[:,:,1:,:,:]-v[:,:,:-1,:,:]
        gy = v[:,:,:,1:,:]-v[:,:,:,:-1,:]
        gz = v[:,:,:,:,1:]-v[:,:,:,:,:-1]
        return gx,gy,gz
    gx1,gy1,gz1 = gradient_3d(v1)
    gx2,gy2,gz2 = gradient_3d(v2)
    gx1 = F.pad(gx1,(0,0,0,0,1,0))
    gy1 = F.pad(gy1,(0,0,0,1,0,0))
    gz1 = F.pad(gz1,(0,0,1,0,0,0))
    gx2 = F.pad(gx2,(0,0,0,0,1,0))
    gy2 = F.pad(gy2,(0,0,0,1,0,0))
    gz2 = F.pad(gz2,(0,0,1,0,0,0))
    n1 = torch.sqrt(gx1**2+gy1**2+gz1**2)+1e-8
    n2 = torch.sqrt(gx2**2+gy2**2+gz2**2)+1e-8
    nx1 = gx1/n1; ny1 = gy1/n1; nz1 = gz1/n1
    nx2 = gx2/n2; ny2 = gy2/n2; nz2 = gz2/n2
    dot = nx1*nx2+ny1*ny2+nz1*nz2
    mask = ((v1>0.5)|(v2>0.5)).float()
    mean_dot = (dot*mask).sum()/(mask.sum()+1e-8)
    return mean_dot.item()
