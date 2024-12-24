"""
Visualization tools for NeuroHoloNet.
"""

import os
import matplotlib
import matplotlib.pyplot as plt

class AttentionVisualizer:
    def __init__(self, out_dir="results/figures"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        matplotlib.rcParams['figure.dpi'] = 300

    def visualize_3d_attention(self, attn_3d, prefix="attn"):
        """Visualize a slice of 3D attention map."""
        mid = attn_3d.shape[0]//2
        atslice = attn_3d[mid]
        plt.figure()
        plt.imshow(atslice, cmap='hot')
        plt.title("3D Attention (mid-slice)")
        path = os.path.join(self.out_dir, f"{prefix}_slice.png")
        plt.savefig(path, dpi=300)
        plt.close()

    def compare_baselines(self, neu_img, baseline_img, prefix="compare"):
        """Compare NeuroHolo output with baseline."""
        fig, axs = plt.subplots(1,2, figsize=(6,3))
        axs[0].imshow(neu_img, cmap='gray')
        axs[0].set_title("NeuroHolo")
        axs[1].imshow(baseline_img, cmap='gray')
        axs[1].set_title("Baseline")
        plt.tight_layout()
        path = os.path.join(self.out_dir, f"{prefix}_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
