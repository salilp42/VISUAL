"""
Loss functions for training NeuroHoloNet.
"""

import torch
import torch.nn as nn
from ..utils.geometry import compute_curvature, compute_topology_measure

class CurvatureConsistencyLoss(nn.Module):
    def forward(self, curv_map):
        return curv_map.mean()

class TopologyConsistencyLoss(nn.Module):
    def forward(self, topo_vals):
        return topo_vals.mean()*0.1

class NeuroHoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.curv = CurvatureConsistencyLoss()
        self.topo = TopologyConsistencyLoss()

    def forward(self, logits, labels, input_voxels):
        cls_loss = self.ce(logits, labels)
        c_map = compute_curvature(input_voxels)
        c_loss = self.curv(c_map)
        t_vals = compute_topology_measure(input_voxels)
        t_loss = self.topo(t_vals.unsqueeze(1))
        total = cls_loss + 0.01*c_loss + 0.01*t_loss
        return total
