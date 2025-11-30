# DiffScale

**Synthetic Code Scale Generation via Conditional Diffusion Models for Robust Angular Sensing**

DiffScale is a deep learning framework designed to generate synthetic images of optical code scales used in digital angle transducers. The system leverages conditional diffusion models to produce both ideal and realistically distorted image sequences (200×4 pixels), enabling robust training and evaluation of angular reading algorithms under adverse conditions typical of robotic applications.

## Overview

Digital angle transducers based on pseudo-random and incremental code scales are critical components in precision motion control systems. However, their reliability is often compromised by optical distortions such as motion blur, sensor noise, scratches, dust spots, and non-uniform illumination.

DiffScale addresses this challenge by:

- Generating physically consistent, temporally coherent sequences of code scale images that simulate continuous rotation.

- Modeling static sensor-level defects (e.g., scratches or spots) that remain fixed in pixel coordinates across frames, reflecting real-world acquisition conditions.

- Providing a scalable, controllable, and fully synthetic data source that eliminates the need for costly and time-consuming physical data collection.

## Key Features

- **Conditional Diffusion Model**: A U-Net-based architecture conditioned on scale type (incremental or PRBS), distortion category, and spatial offset, enabling precise control over generated samples.

- **Coherent Sequence Generation**: Each frame in a sequence is generated with a deterministic phase increment (`offset_t = offset_{t-1} + Δ`), ensuring temporal continuity.

- **Realistic Distortion Modeling**: Supports multiple distortion types, including:

    - Blur (motion),
    - Gaussian noise,
    - Scratches (simulated as localized intensity reductions),
    - Lighting non-uniformity,
    - Spot artifacts (dust or sensor defects),
    - Combined and custom distortions.

- **Reproducible Pipeline**: Configuration-driven via `configs/Diffusion.yaml`.
- **Secure Inference**: Weights are loaded with `weights_only=True`.

## Project Structure

```
DiffScale/
├── configs/              # Model and training hyperparameters
├── data/                 # Generated datasets (excluded from Git)
│   ├── clean/            # Ideal (distortion-free) scale images
│   ├── distorted/        # Realistically distorted images
│   └── metadata.csv      # Annotations: sequence_id, frame_id, offset, distortion_type, paths
├── src/
│   ├── data_gen.py       # Synthetic data generator with coherent sequences
│   ├── dataset.py        # PyTorch Dataset with conditional labels
│   ├── model.py          # Conditional diffusion model and U-Net
│   ├── train.py          # Training loop with checkpointing
│   └── inference.py      # Sample generation from trained model
├── outputs/              # Model weights and generated samples
├── checkpoints/          # Training checkpoints (optimizer state, epoch, etc.)
├── requirements.txt      # Python dependencies
└── README.md
```

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Generate dataset (ideal + distorted sequences):
```bash
python -m src.data_gen
```

Train the diffusion model:
```bash
python -m src.data_gen
```

Generate synthetic samples:
```bash
python -m src.inference
```

All hyperparameters (image size, distortions, timesteps, network depth) are configurable in `configs/diffusion_config.yaml`.
