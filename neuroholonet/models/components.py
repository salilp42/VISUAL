"""
Neural network components for the DRHPNet architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from ..utils.geometry import compute_curvature, compute_topology_measure

class AdaptiveFoveation(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.size = size
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        half = self.size/2.0
        zc, yc, xc = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
        )
        dist = ((xc-half)**2 + (yc-half)**2 + (zc-half)**2).sqrt()
        dist = dist/dist.max()
        dist = dist.to(x.device)
        w = torch.exp(-self.gamma*dist)
        w = w.unsqueeze(0)
        return x*w

class Small3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class EnhancedGeometricModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = GeometricGate(in_dim=2, hidden_dim=16)
        
    def forward(self, x):
        if x.shape[1] > 1:
            xm = x[:,0:1].mean(dim=1, keepdim=True)
        else:
            xm = x
        c = compute_curvature(xm)
        t = compute_topology_measure(xm)
        m = self.gate(c, t)
        return m

class GeometricGate(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, curv, topo):
        B = curv.shape[0]
        cm = curv.mean(dim=[1,2,3,4])
        tv = topo.squeeze(1)
        comb = torch.stack([cm, tv], dim=-1)
        raw = self.mlp(comb)
        gate = torch.sigmoid(raw)
        D, H, W = curv.shape[-3:]
        gmap = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B,1,D,H,W)
        return gmap

class ContinuousOrientationEncoder(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, coords):
        return self.mlp(coords)

class DRHPAttention(nn.Module):
    def __init__(self, in_channels=16, d_model=32, ori_dim=16):
        super().__init__()
        self.d_model = d_model
        self.ori_dim = ori_dim
        self.scale1 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.scale2 = nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1)
        self.orien = ContinuousOrientationEncoder(embed_dim=ori_dim)
        self.Wq = nn.Linear(in_channels+ori_dim, d_model)
        self.Wk = nn.Linear(in_channels+ori_dim, d_model)
        self.Wv = nn.Linear(in_channels, d_model)
        self.geo = EnhancedGeometricModule()
        self.prune_th = 1e-3

    def forward(self, feat_map):
        B, C, D, H, W = feat_map.shape
        s1 = self.scale1(feat_map)
        s2 = self.scale2(feat_map)
        s2_up = F.interpolate(s2, size=(D,H,W), mode='trilinear', align_corners=False)
        mf = 0.5*s1 + 0.5*s2_up
        mf = torch.where(mf.abs()<self.prune_th, torch.zeros_like(mf), mf)
        N = D*H*W
        ff = mf.view(B,C,N).permute(0,2,1)
        
        zc, yc, xc = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
        )
        coords = torch.stack([xc,yc,zc], dim=-1).float().to(feat_map.device).view(-1,3)
        oe = self.orien(coords).unsqueeze(0).expand(B,-1,-1)
        
        qk = torch.cat([ff,oe], dim=-1)
        Q = self.Wq(qk)
        K = self.Wk(qk)
        V = self.Wv(ff)
        
        gm3d = self.geo(mf)
        gf = gm3d.view(B,-1)
        gq = gf.unsqueeze(-1)
        gk = gf.unsqueeze(1)
        gp = gq*gk
        
        def _attention_block(Q, K, V, d_model, gate_pair):
            scores = torch.matmul(Q, K.transpose(1,2))/math.sqrt(d_model)
            attn = F.softmax(scores, dim=-1)
            kfrac = 0.3
            B, N, _ = attn.shape
            tk = int(kfrac*N)
            vals, idxs = torch.topk(attn, tk, dim=-1)
            mask = torch.zeros_like(attn)
            mask.scatter_(-1, idxs, 1.0)
            attn = attn*mask
            attn = attn*gate_pair
            attn = attn/(attn.sum(dim=-1,keepdim=True)+1e-8)
            out = torch.matmul(attn, V)
            return out
            
        def _ckpt_forward(aQ, aK, aV, dm, gp_):
            return _attention_block(aQ, aK, aV, dm, gp_)
            
        out_tokens = cp.checkpoint(_ckpt_forward, Q, K, V,
                                 torch.tensor(self.d_model, device=Q.device), gp)
        out_3d = out_tokens.permute(0,2,1).view(B, self.d_model, D, H, W)
        return out_3d

class UncertaintyRouter(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1), nn.ReLU(),
            nn.Conv3d(in_channels, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class FeatureRefinementModule(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, feats, feedback, unc):
        alpha = 1.0-unc
        return feats + alpha*self.conv(feedback)

class EnhancedFeedbackModule(nn.Module):
    def __init__(self, magno_channels=8, parvo_channels=8):
        super().__init__()
        self.unc_mag = UncertaintyRouter(magno_channels)
        self.ref_mag = FeatureRefinementModule(magno_channels)
        self.unc_par = UncertaintyRouter(parvo_channels)
        self.ref_par = FeatureRefinementModule(parvo_channels)
        
    def forward(self, magno, parvo, fbmag, fbpar):
        um = self.unc_mag(magno)
        mr = self.ref_mag(magno, fbmag, um)
        up = self.unc_par(parvo)
        pr = self.ref_par(parvo, fbpar, up)
        return mr, pr
