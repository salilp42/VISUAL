"""
Core DRHPNet model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .components import (
    AdaptiveFoveation,
    Small3DConv,
    DRHPAttention,
    EnhancedFeedbackModule
)

class DRHPNet(nn.Module):
    def __init__(self, voxel_size=32, base_channels=8, d_model=32, num_classes=2):
        super().__init__()
        self.size = voxel_size
        self.retina = AdaptiveFoveation(size=voxel_size)
        self.magno = Small3DConv(1, base_channels, stride=2)
        self.parvo = Small3DConv(1, base_channels, stride=1)
        self.merge = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        self.drhp = DRHPAttention(in_channels=base_channels*2, d_model=d_model, ori_dim=16)
        self.fb = EnhancedFeedbackModule(magno_channels=base_channels, parvo_channels=base_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_fov = self.retina(x)
        mf = self.magno(x_fov)
        pf = self.parvo(x_fov)
        Bm, Cm, Dm, Hm, Wm = mf.shape
        mf_up = F.interpolate(mf, size=(Dm*2, Hm*2, Wm*2), mode='trilinear', align_corners=False)
        merged = torch.cat([mf_up, pf], dim=1)
        merged = self.merge(merged)
        drhp_out = self.drhp(merged)
        fb_mag = drhp_out[:, :Cm]
        fb_par = drhp_out[:, Cm:Cm*2]
        fb_mag_dn = F.interpolate(fb_mag, size=(Dm, Hm, Wm), mode='trilinear', align_corners=False)
        _, _ = self.fb(mf, pf, fb_mag_dn, fb_par)
        logits = self.classifier(drhp_out)
        return logits
