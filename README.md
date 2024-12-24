# NeuroHoloNet
A neural network architecture for 3D voxel-based shape analysis, drawing inspiration from the dual-stream hypothesis of early visual processing—namely, magnocellular (dorsal) and parvocellular (ventral) pathways—as well as established principles of recurrent and feedback connections in the primate visual cortex. The method incorporates geometric and topological consistency measures to enhance robustness when modeling complex shapes.

## Overview
NeuroHoloNet is designed for volumetric (voxel-based) shape analysis and classification tasks. The approach combines:

### Dual-Stream Processing
Based on conceptual analogies to magnocellular (motion, low spatial resolution) and parvocellular (high spatial resolution, color) pathways 1,2. Each stream is specialized for different aspects of the input volume and later fused.

### Adaptive Foveation
Mimics the eye's foveal mechanism by focusing higher resolution on critical regions. This mechanism is adjustable via learned parameters for more biologically consistent spatial weighting.

### Dynamic Geometric Gating
Incorporates curvature and topological measures 3,4, aligning with the notion that higher-order cortical areas modulate early processing streams through feedback 5,6. Curvature and connected-component estimates guide selective feature refinement.

### Feedback Refinement
Draws from evidence of extensive cortical feedback loops 5,6. After a feedforward pass, feature maps are updated by an uncertainty-driven router that refines representation quality.

### Extensive Data Augmentation
Employs synthetic transformations, including rotations, scalings, surface perturbations, and occlusions, to bolster generalization.

## Key Features
- Biologically-Inspired Dual Streams: Separate magno and parvo-like 3D convolutions capture broad motion-sensitive versus fine detail representations.
- Geometric Consistency: Local curvature maps and connected-components-based topology estimates restrict feature collapse and improve structural fidelity.
- Adaptive Foveation: Learned weighting highlights central or relevant regions, analogous to the varying resolution of the retina.
- Feedback Mechanism: A dedicated feedback pathway adjusts uncertain features, reflecting recurrent loops found in the visual cortex.
- 3D Attention Module: A multi-scale attention component with orientation encoding for robust shape discrimination.
- Modular Design: Easily extensible to other 3D network backbones or additional geometric constraints.


## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
VISUAL/
├── neuroholonet/
│   ├── models/          # Neural network architectures
│   ├── data/           # Dataset and data loading utilities
│   ├── utils/          # Helper functions and utilities
│   ├── metrics/        # Evaluation metrics
│   └── visualization/  # Visualization tools
├── tests/             # Unit tests
├── examples/          # Usage examples
└── scripts/          # Training and evaluation scripts
```

## Usage

Basic example:

```python
from neuroholonet.models import DRHPNet
from neuroholonet.data import CubeSphereDataset

# Initialize model
model = DRHPNet(voxel_size=32, base_channels=8)

# Load data
dataset = CubeSphereDataset(n_samples=64, size=32)

# Train and evaluate
# See examples/train.py for full training pipeline
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{patel2024neuroholonet,
  title={NeuroHoloNet: Biologically-Inspired Neural Architecture for 3D Shape Analysis},
  author={Patel, Salil},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Author

Salil Patel - [GitHub](https://github.com/salilp42)
