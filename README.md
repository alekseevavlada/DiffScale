# DiffScale

**Synthetic Code Scale Generation via Conditional Diffusion Models for Robust Angular Sensing**

DiffScale is a deep learning framework designed to generate synthetic images of optical code scales used in digital angle transducers. The system leverages conditional diffusion models to produce both ideal and realistically distorted image sequences (800x4 pixels), enabling robust training and evaluation of angular reading algorithms under adverse conditions typical of robotic applications.

## Overview

Digital angle transducers based on pseudo-random and incremental code scales are critical components in precision motion control systems. However, their reliability is often compromised by optical distortions such as motion blur, sensor noise, scratches, dust spots, and non-uniform illumination.

DiffScale addresses this challenge by:

- Generating physically consistent, temporally coherent sequences of code scale images that simulate continuous rotation.

- Modeling static sensor-level defects (e.g., scratches or spots) that remain fixed in pixel coordinates across frames, reflecting real-world acquisition conditions.

- Providing a scalable, controllable, and fully synthetic data source that eliminates the need for costly and time-consuming physical data collection.

## Subpixel Displacement Estimation (`Subpixel.ipynb`)

This repository includes a comprehensive study on **subpixel displacement estimation** using the generated datasets. The notebook `Subpixel.ipynb` implements and evaluates a pipeline for achieving nanometer-level resolution in tracking moving fringe patterns under severe distortions.

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
├── DiffScale-main/           # Main project source code
│   ├── configs/              # Model and training hyperparameters
│   ├── data/                 # Generated datasets (excluded from Git)
│   │   ├── clean/            # Ideal (distortion-free) scale images
│   │   ├── distorted/        # Realistically distorted images
│   │   └── metadata.csv      # Annotations: sequence_id, frame_id, offset, distortion_type, paths
│   ├── src/
│   │   ├── data_gen.py       # Synthetic data generator with coherent sequences
│   │   ├── dataset.py        # PyTorch Dataset with conditional labels
│   │   ├── model.py          # Conditional diffusion model and U-Net
│   │   ├── train.py          # Training loop with checkpointing
│   │   └── inference.py      # Sample generation from trained model
│   ├── outputs/              # Model weights and generated samples
│   ├── checkpoints/          # Training checkpoints (optimizer state, epoch, etc.)
│   └── requirements.txt      # Python dependencies for the core framework
├── Subpixel.ipynb            # Main research notebook for subpixel displacement estimation
├── requirements.txt          # Python dependencies for the entire project (including Subpixel analysis)
└── README.md                 # This file
```

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Navigate to the main project directory:
```bash
cd DiffScale-main
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

Run the subpixel analysis:
Open and execute `Subpixel.ipynb` in Jupyter Lab or Google Colab.

All hyperparameters (image size, distortions, timesteps, network depth) are configurable in `DiffScale-main/configs/diffusion_config.yaml`.

## References

1. Zhao M, Yuan Y, Luo L, Li X. A Review: Absolute Linear Encoder Measurement Technology. Sensors. 2025; 25(19):5997. https://doi.org/10.3390/s25195997
2. Jianxiang Liao, Xin Chen, Xindu Chen, Fangjian Zhang, and Han Wang "High speed image acquisition system of absolute encoder", Proc. SPIE 10322, Seventh International Conference on Electronics and Information Engineering, 103221C (23 January 2017); https://doi.org/10.1117/12.2265226
3. Shi, Boxin & Zhao, Hang & Ben-Ezra, Moshe & Yeung, Sai-Kit & Fernandez-Cull, Christy & Shepard, Hamilton & Barsi, Christopher & Raskar, Ramesh. (2015). Sub-pixel Layout for Super-Resolution with Images in the Octic Group. 8689. 10.1007/978-3-319-10590-1_17. 
4. Chang, Li & Xu, Hui & Liu, Ben & Li, Jian. (2010). All Digital Nanometer Subdivision Method to Process Coarse Grating Signal. Advanced Materials Research. 108-111. 1199-1204. 10.4028/www.scientific.net/AMR.108-111.1199. 
5. Joseph D. Tobiason, Avron M. Zwilling, Casey E. Emtman. Compact Nanometer Resolution Fiber-optic Encoder with
Embedded Reference Mark. Proceedings of the euspen International Conference, Delft (June 2010). 
6. RESOLUTE optical absolute encoder series. https://www.renishaw.com/en/resolute-optical-absolute-encoder-series--37823?srsltid=AfmBOoqRuim0xJJDwzKioLOukfeyhXfnIczSu8chxkVGeBxQd_XaOkuh
