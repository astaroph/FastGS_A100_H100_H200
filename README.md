<div align="center">
<h1>FastGS: Training 3D Gaussian Splatting in 100 Seconds</h1> 
<h2>CVPR 2026</h2> 

[🌐 Homepage](https://fastgs.github.io/) | [📄 Paper](https://arxiv.org/abs/2511.04283) ｜[🤗 Pre-trained model](https://huggingface.co/Goodsleepeverday/fastgs)

</div>


## Fork of Original FastGS repo with custom Ampere and Hopper Class GPU build instructions, intended for Slurm cluster environment
### Credit and upstream provenance
FastGS is an external project and should be credited to its original authors and repository. Original links to this repo have been retained above

## Build notes
In this build workflow, a user may choose to use either the original FastGS repository or a fork that contains local compatibility fixes and pipeline-specific adjustments.

The build recipes below were adapted from successful installs used in this pipeline environment, but they are not guaranteed to work unchanged on every cluster. In particular, users must verify their own module names, compiler toolchains, CUDA installation paths, and available PyTorch wheel compatibility before building.

The successful build recipes below used:

* Python: `3.10.13`
* GCC: `10.2.0`
* CUDA module: `11.8.0`

These exact names may differ on your cluster even when the underlying versions are the same.

### Important cluster note

Before running either build route below, verify all of the following on your own system:

* the correct module names for `gcc` and `cuda`
* the correct path to your conda or mamba installation
* the correct CUDA root path for `CUDA_HOME`
* whether your cluster permits internet access during environment creation
* whether your GPU nodes support the PyTorch and CUDA wheel combination you are installing

The commands below use placeholder paths such as:

* `PATH/TO/USER/SOFTWARE`
* `PATH/TO/USER/CONDA_ENVS`
* `PATH/TO/USER/CONDA_PKGS`
* `PATH/TO/CUDA`

Replace all of them with real locations before running.

---

## A100 build route

This route was used successfully for A100-class GPUs.

### 1. Start from a clean shell and load modules

```bash
module purge
module load gcc/10.2.0
module load cuda/11.8.0

conda deactivate 2>/dev/null || true
```

### 2. Set user-specific locations

```bash
export REPO_PARENT="PATH/TO/USER/SOFTWARE"
export REPO_DIR="${REPO_PARENT}/FastGS"
export ENV_DIR="PATH/TO/USER/CONDA_ENVS/FastGS"
export CONDA_PKGS_DIRS="PATH/TO/USER/CONDA_PKGS"

mkdir -p "${REPO_PARENT}"
mkdir -p "${CONDA_PKGS_DIRS}"
```

### 3. Clone FastGS

```bash
cd "${REPO_PARENT}"
git clone --recursive https://github.com/astaroph/FastGS_A100_H100_H200.git
cd "${REPO_DIR}"
```

### 4. Create a minimal conda environment

```bash
mamba create -y \
  -p "${ENV_DIR}" \
  python=3.10.13 \
  pip=22.3.1 \
  plyfile \
  tqdm \
  websockets \
  "numpy<2" \
  "setuptools<81" \
  -c conda-forge -c pytorch
```

### 5. Activate the environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
```

### 6. Install CUDA-enabled PyTorch

This successful recipe used PyTorch `1.12.1+cu116`, even though the cluster CUDA module loaded was `11.8.0`. That worked in the original tested environment, but users should verify compatibility for their own systems.

```bash
python -m pip install \
  torch==1.12.1+cu116 \
  torchvision==0.13.1+cu116 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

### 7. Set build environment variables

```bash
export CUDA_HOME="PATH/TO/CUDA"
export CC="$(which gcc)"
export CXX="$(which g++)"
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export MAX_JOBS=4
```

### 8. Install FastGS submodules

```bash
pip install ./submodules/diff-gaussian-rasterization_fastgs
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

### 9. Sanity check

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
PY
```

---

## H100 / H200 build route

This route was used successfully for Hopper-class systems, including H100 and H200.

### 1. Start from a clean shell and load modules

```bash
module purge
module load gcc/10.2.0
module load cuda/11.8.0

conda deactivate 2>/dev/null || true
```

### 2. Set user-specific locations

```bash
export REPO_PARENT="PATH/TO/USER/SOFTWARE"
export REPO_DIR="${REPO_PARENT}/FastGS"
export ENV_DIR="PATH/TO/USER/CONDA_ENVS/FastGS_h100"
export CONDA_PKGS_DIRS="PATH/TO/USER/CONDA_PKGS"

mkdir -p "${REPO_PARENT}"
mkdir -p "${CONDA_PKGS_DIRS}"
```

### 3. Clone FastGS if needed

```bash
if [[ ! -d "${REPO_DIR}" ]]; then
  cd "${REPO_PARENT}"
  git clone --recursive https://github.com/astaroph/FastGS_A100_H100_H200.git
fi

cd "${REPO_DIR}"
```

### 4. Create the conda environment

```bash
mamba create -y -p "${ENV_DIR}" \
  python=3.10.13 \
  pip \
  "setuptools<81" \
  wheel \
  ninja \
  cmake \
  packaging \
  "numpy<2" \
  pillow \
  plyfile \
  tqdm \
  websockets
```

### 5. Activate the environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
```

### 6. Install CUDA-enabled PyTorch

This successful recipe used PyTorch `2.4.1+cu118`.

```bash
python -m pip install \
  torch==2.4.1+cu118 \
  torchvision==0.19.1+cu118 \
  torchaudio==2.4.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 7. Set build environment variables

```bash
export CUDA_HOME="PATH/TO/CUDA"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CC="$(which gcc)"
export CXX="$(which g++)"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
```

### 8. Verify PyTorch sees the GPU

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
PY
```

### 9. Install FastGS submodules

For this successful build, the submodules were installed with `--no-build-isolation`.

```bash
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_fastgs
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/fused-ssim
```

### 10. Import test

```bash
python - <<'PY'
import torch
import diff_gaussian_rasterization_fastgs
import simple_knn
import fused_ssim
print("All core imports succeeded.")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## Required FastGS source modification for this pipeline

For this pipeline’s COLMAP dataset layout, FastGS must preserve subdirectory information when resolving image paths. If your FastGS checkout still uses `os.path.basename(extr.name)` inside `scene/dataset_readers.py`, update it as follows.

### Change

```python
# from
image_path = os.path.join(images_folder, os.path.basename(extr.name))

# to
image_path = os.path.join(images_folder, extr.name)
```

### Why this is required

This pipeline may stage COLMAP-compatible image datasets with camera subdirectories inside the image root. In that case, stripping the basename causes FastGS to lose the relative camera-subdirectory structure and fail to resolve images correctly.

---


## Troubleshooting

### Build succeeds but import fails

Check that:

* `CUDA_HOME` points to the correct toolkit root
* `gcc` and `g++` resolve to the intended compiler
* the environment was activated before installing submodules
* your PyTorch CUDA wheel matches the intended GPU architecture closely enough for your cluster

### Dataset loads fail even though training starts

Check `scene/dataset_readers.py` and confirm the `extr.name` change above is present.

### A100 and Hopper builds behave differently

That is expected in this workflow. The successful recipes used different PyTorch versions and different architecture settings:

* A100 recipe: `TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"`
* H100/H200 recipe: `TORCH_CUDA_ARCH_LIST="9.0"`
<div align="center">




<p align="center">
    <img src="assets/teaser_fastgs.png" width="800px"/>
</p>

## 🚀 What Makes FastGS Special?

FastGS is a **general acceleration framework** that supercharges 3D Gaussian Splatting training while maintaining Comparable rendering quality. Our method stands out with:

- **⚡ Blazing Fast Training**: Achieve SOTA results within **100 seconds**. **3.32× faster** than DashGaussian on Mip-NeRF 360 dataset. **15.45× acceleration** vs vanilla 3DGS on Deep Blending.
- **⚡ High fidelity**: Comparable rendering quality with SOTA methods
- **🎯 Easy Integration**: Seamlessly integrates with various backbones (Vanilla 3DGS, Scaffold-GS, Mip-splatting, etc.)
- **🛠️ Multi-Task Ready**: Proven effective across dynamic scenes, surface reconstruction, sparse-view, large-scale, and SLAM tasks
- **💡 Memory-Efficient**: Low GPU Memory requirements make it accessible for various hardware setups
- **🔧 Easy Deployment**: Simple post-training tool for feedforward 3DGS that works out-of-the-box

## 📢 Latest Updates
#### 🔥 **[2026.03]** The surface reconstruction code [Fast-PGSR](https://github.com/fastgs/FastGS/tree/fast-pgsr) has been released!
#### 🎉 **[2026.02]** Our work has been accepted to CVPR 2026! 🤗🤗🤗
#### 🥇 **[2026.01]** Our method was used as a component in the [winning solution](https://arxiv.org/pdf/2601.19489) (1st place🥇) of the **[SIGGRAPH Asia 2025 3DGS Fast Reconstruction Challenge](https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/index.html#awards)**. We sincerely thank the **3DV-CASIA** for their interest and adoption of our work.
#### 🔥 **[2025.12.03]** The sparse-view reconstruction code [Fast-DropGaussian](https://github.com/fastgs/FastGS/tree/fast-dropgaussian) has been released!
#### 🔥 **[2025.11.29]** The dynamic scene reconstruction code [Fast-D3DGS](https://github.com/fastgs/FastGS/tree/fast-d3dgs) has been released!
#### 🔧 **[2025.11.27]** The tutorial has been released — see the [Wiki](https://github.com/fastgs/FastGS/wiki)!
#### 📄 **[2025.11.26]** The supplementary material has been released [here](https://arxiv.org/abs/2511.04283)!
#### 🔥 **[2025.11.17]** Pre-trained model Released 🤗!
#### 🔥 **[2025.11.16]** Code Released - Get Started Now! 🚀


## 🎯 Coming Soon

#### Released Modules
- **Dynamic Scenes Reconstruction** — [Fast-D3DGS](https://github.com/fastgs/FastGS/tree/fast-d3dgs) (based on [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)) — Released
- **Sparse-view Reconstruction** — [Fast-DropGaussian](https://github.com/fastgs/FastGS/tree/fast-dropgaussian) (based on [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release)) — Released 
- **Surface Reconstruction** — [Fast-PGSR](https://github.com/fastgs/FastGS/tree/fast-pgsr) (based on [PGSR](https://github.com/zju3dv/PGSR)) — Released 

#### To Be Released
- **Backbone Enhancing** — based on [Mip-splatting](https://github.com/autonomousvision/mip-splatting)
- **SLAM** — based on [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) 
- **Autonomous Driving Scenes** — based on [street_gaussians](https://github.com/zju3dv/street_gaussians)
- **Large-scale Reconstruction** — based on [OctreeGS](https://github.com/city-super/Octree-GS/tree/main)

## 🏗️ Training Framework

Our training pipeline leverages **PyTorch** and optimized **CUDA extensions** to efficiently produce high-quality trained models in record time.

### 💻 Hardware Requirements

- **GPU**: CUDA-ready GPU with Compute Capability 7.0+
- **Memory**: 24 GB VRAM (for paper-quality results; we recommend NVIDIA RTX4090)

### 📦 Software Requirements

- **Conda** (recommended for streamlined setup)
- **C++ Compiler** compatible with PyTorch extensions
- **CUDA SDK 11** (or compatible version)
- **⚠️ Important**: Ensure C++ Compiler and CUDA SDK versions are compatible

### ⚠️ CUDA Version Reference

Our testing environment uses the following CUDA configuration:

| Component                             | Version          |
|---------------------------------------|------------------|
| Conda environment CUDA version        | 11.6             |
| Ubuntu system `nvidia-smi` CUDA       | 12.2             |
| `nvcc -V` compiler version            | 11.8 (v11.8.89)  |

> **Note**: The Conda CUDA and system CUDA versions may differ. The compiler version (`nvcc`) is what matters for PyTorch extensions compilation (diff-gaussian-rasterization_fastgs).


## 🚀 Quick Start

### 📥 Clone the Repository

```bash
git clone https://github.com/fastgs/FastGS.git --recursive
cd FastGS
```

### ⚙️ Environment Setup

We provide a streamlined setup using Conda:

```shell
# Windows only
SET DISTUTILS_USE_SDK=1

# Create and activate environment
conda env create --file environment.yml
conda activate fastgs
```

### 📂 Dataset Organization

Organize your datasets in the following structure:

```bash
datasets/
├── mipnerf360/
│   ├── bicycle/
│   ├── flowers/
│   └── ...
├── db/
│   ├── playroom/
│   └── ...
└── tanksandtemples/
    ├── truck/
    └── ...
```

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). 

## 🎯 Training & Evaluation

### ⚡ FastGS (Standard)

Train the base model with optimal speed and quality balance:

```bash
bash train_base.sh
```

### 🎨 FastGS-Big (High Quality)

For enhanced quality with slightly longer training time:

```bash
bash train_big.sh
```
<details>
<summary><span style="font-weight: bold;">📋 Advanced: Command Line Arguments for train.py</span></summary>

  #### --loss_thresh
  Threshold of the loss map; a lower value generally results in more Gaussians being retained.
  #### --grad_abs_thresh 
  Absolute gradient (same as Abs-GS) threshold for split.
  #### --grad_thresh
  Gradient(same as vanilla 3DGS) threshold for clone.
  #### --highfeature_lr
  Learning rate for high-order SH coefficients (features_rest).
  #### --lowfeature_lr
  Learning rate for low-order SH coefficients (features_dc).
  #### --dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified.
  #### --mult 
  Multiplier for the compact box to control the tile number of each splat
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

Note that similar to MipNeRF360 and vanilla 3DGS, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.

## 🎬 Interactive Viewers

Our 3DGS representation is identical to vanilla 3DGS, so you can use the official [SIBR viewer](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) for interactive visualization. For a quick start without local setup, try the web-based [Supersplat](https://superspl.at/editor).

## 🎯 Quick Facts

| Feature | FastGS | Previous Methods |
|---------|---------|---------------------|
| Training Time | **100 seconds** | 5-30 minutes |
| Gaussian Efficiency | ✅ **Strict Control** | ❌ Redundant Growth |
| Memory Usage | ✅ **Low Footprint** | ❌ High Demand |
| Task Versatility | ✅ **6 Domains** | ❌ Limited Scope |

## 📧 Contact

If you have any questions, please contact us at **renshiwei@mail.nankai.edu.cn**.


## 🙏 Acknowledgements

This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Taming-3DGS](https://github.com/humansensinglab/taming-3dgs), [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat), and [Abs-GS](https://github.com/TY424/AbsGS). We extend our gratitude to all the authors for their outstanding contributions and excellent repositories!

**License**: Please adhere to the licenses of 3DGS, Taming-3DGS, and Speedy-Splat.

Special thanks to the authors of [DashGaussian](https://github.com/YouyuChen0207/DashGaussian) for their generous support!


## Citation
If you find this repo useful, please cite:
```
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}

```

---

<div align="center">

**⭐ If FastGS helps your research, please consider starring this repository!**

*FastGS: Training 3D Gaussian Splatting in 100 Seconds*

</div>

---
