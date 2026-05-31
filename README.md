# MTSIR3-GAN: Multivariate Time Series Imputation using R3GAN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文文档](README_CN.md) | English

## 📖 Overview

MTSIR3-GAN is a research project adapting **R3GAN** (NeurIPS 2024) from image generation to **Multivariate Time Series Imputation (MTSI)**. This repository contains both the original thesis implementation and an extended empirical study (FMGAN) examining when adversarial refinement helps for time series imputation.

### Key Features

- 🏗️ **Modern Architecture**: R3GAN-1D adaptation with 1D convolutions, Fixup initialization, and a frequency-domain discriminator
- 🔬 **Robust Training**: Regularized relativistic loss (RpGAN + R₁ + R₂) for stable adversarial training
- 🌐 **Interactive GUI**: Dash-based web interface for data upload, visualization, and imputation
- 📊 **Extended Empirical Study (FMGAN)**: a systematic study of coarse-to-fine adversarial refinement across multiple datasets and configurations

> MS-thesis results comparing MTSIR3-GAN against SSGAN and TimesNet are summarized under [Experimental Results](#-experimental-results). They originate from the thesis evaluation and are reproducible via the training scripts below.

### Extended Study: FMGAN

The [`FMGAN/`](FMGAN/) directory contains a follow-up empirical study of coarse-to-fine adversarial refinement for time series imputation. A detailed write-up is forthcoming.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/he-yufeng/MTSIR3-GAN.git
cd MTSIR3-GAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# For the FMGAN empirical study, also install its extras
pip install -r FMGAN/requirements.txt
```

### Launch the GUI

```bash
cd PURE-GUIv2.0
python app_dad.py
```

Then open your browser and navigate to `http://127.0.0.1:8050`

> **Note:** Pretrained weights (`.pth`) and cached arrays are **not bundled** in the repository to keep it lightweight. Train a model first (see [Training Models from Scratch](#2-training-models-from-scratch)), or place your own weights under `PURE-GUIv2.0/model_files/`.

## 🎮 Usage

### 1. Web Interface (Recommended for Beginners)

The interactive GUI provides three main modules:

- **Data Analysis**: Upload and visualize your time series data, analyze missing patterns
- **Data Imputation**: Apply MTSIR3-GAN, SSGAN, or TimesNet for imputation
- **Model Visualization**: Compare model performance and visualize results

### 2. Training Models from Scratch

#### Train MTSIR3-GAN (R3GAN)

```bash
cd R3GAN

# Train on AirQuality dataset
python train.py --outdir=./training_runs --data=../datasets/AirQuality/pm25_missing.txt \
                --gpus=1 --batch=64 --gamma=0.5 --preset=AirQuality_MTSI

# Train on PSM dataset
python train.py --outdir=./training_runs --data=../datasets/PSM/train.csv \
                --gpus=1 --batch=32 --gamma=1.0 --preset=PSM_MTSI
```

#### Train SSGAN

```bash
cd SSGAN

# Edit main.py to set dataset and parameters
python main.py --epochs=50 --batch_size=64 --model=Based_on_BRITS
```

#### Train TimesNet

```bash
cd TimesNet

# Train for imputation task
python run.py --task_name imputation --data PSM --root_path ./datasets/PSM/ \
              --data_path train.csv --model_id PSM_mask_0.25 --model TimesNet \
              --mask_rate 0.25 --enc_in 25 --dec_in 25 --c_out 25 \
              --batch_size 16 --learning_rate 0.001 --train_epochs 10
```

### 3. Model Inference

```python
import torch
import numpy as np

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_files/AirQuality_R3GAN.pth')
model.eval()

# Prepare your data
incomplete_data = np.load('your_data.npy')  # Shape: (batch, channels, height, width)
mask = np.load('your_mask.npy')  # 1 for observed, 0 for missing

# Generate imputations
with torch.no_grad():
    z = torch.randn(batch_size, noise_dim).to(device)
    condition = torch.from_numpy(incomplete_data * mask).to(device)
    imputed = model(z, condition)

# Combine with observed values
final = incomplete_data * mask + imputed.cpu().numpy() * (1 - mask)
```

## 📁 Project Structure

Each top-level folder has its own `README.md` with usage and provenance.

```
MTSIR3-GAN/
├── R3GAN/                    # Thesis model (R3GAN adapted to MTSI); see R3GAN/README.md
│   ├── train.py              # Training entry point
│   ├── gen_timeseries.py     # Generation / imputation
│   ├── R3GAN/                # Network architectures (relativistic-loss GAN)
│   └── training/             # Training loop and losses
├── SSGAN/                    # SSGAN baseline (Miao et al., AAAI 2021)
├── TimesNet/                 # TimesNet baseline (vendored from THUML TSLib)
├── PURE-GUIv2.0/             # Dash web interface
├── FMGAN/                    # Extended empirical study (coarse-to-fine refinement)
│   ├── models/r3gan_1d.py    # R3GAN-1D architecture (1D adaptation)
│   ├── train_refiner.py      # Coarse-to-fine training script
│   ├── foundation_model/     # MOMENT foundation model wrapper
│   ├── evaluation/           # Metrics, baselines (SAITS/BRITS/CSDI)
│   ├── data/                 # Unified data loading (3 missing patterns)
│   └── scripts/              # Experiment runner scripts
├── datasets/                 # Sample data + download instructions; see datasets/README.md
├── requirements.txt          # Core deps (+ FMGAN/requirements.txt for the study)
└── LICENSE                   # MIT (original code); third-party folders keep their own
```

## 📊 Datasets

The project includes benchmarks on three standard datasets:

1. **PhysioNet Challenge 2012**: ICU patient vital signs (41 features, >80% missing)
2. **Beijing Air Quality**: PM2.5 from 36 stations (36 features, ~13% missing)
3. **Pooled Server Metrics (PSM)**: Server performance metrics (25 features)

### Dataset Download

Due to GitHub's file size limits, large datasets are not included. Download them from:

- **PhysioNet**: https://physionet.org/content/challenge-2012/1.0.0/
- **Air Quality**: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **PSM**: Sample files included in `datasets/PSM/`

Place downloaded datasets in the `datasets/` directory following the structure in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## 🎯 Model Architecture

### MTSIR3-GAN Core Components

1. **Time Series Patching**: Converts 1D sequences (K×L) into 2D patches (C×H×W) for CNN processing
2. **Multi-Matrix Input**: Uses observed data + mask matrix + time-lag matrix as input
3. **Modernized Backbone**: ResNet-style generator and discriminator with:
   - Grouped convolutions and inverted bottlenecks
   - Fixup initialization (no normalization layers)
   - Residual connections for deep networks
4. **Regularized Relativistic Loss**: RpGAN + R₁ + R₂ for stable, diverse generation

```
Generator:
  Input: Noise (z) + Condition (observed data, mask, time-lag)
  ↓
  Embedding Layer (if conditional)
  ↓
  Stage 1: [Upsample → ResBlock × N]
  Stage 2: [Upsample → ResBlock × N]
  Stage 3: [Upsample → ResBlock × N]
  Stage 4: [Upsample → ResBlock × N]
  ↓
  Aggregation Layer (1×1 Conv)
  ↓
  Output: Imputed patch (K×H×W)

Discriminator:
  Input: Data patch (real/fake)
  ↓
  Extraction Layer (1×1 Conv)
  ↓
  Stage 1: [ResBlock × N → Downsample]
  Stage 2: [ResBlock × N → Downsample]
  Stage 3: [ResBlock × N → Downsample]
  Stage 4: [ResBlock × N → Global Pool]
  ↓
  Conditional Projection (if conditional)
  ↓
  Output: Realness logit
```

## 📈 Experimental Results

MS-thesis evaluation — imputation error (MAE ↓ is better) on benchmark datasets. These numbers are from the original thesis study; rerun the training scripts to reproduce them.

| Dataset      | TimesNet  | SSGAN     | MTSIR3-GAN |
|--------------|-----------|-----------|------------|
| AirQuality   | **0.396** | 0.435     | 0.412      |
| PhysioNet    | 0.656     | **0.598** | 0.631      |
| PSM (12.5%)  | 0.544     | 0.586     | **0.524**  |
| PSM (25%)    | **0.649** | 0.683     | 0.671      |
| PSM (50%)    | 0.782     | 0.761     | **0.737**  |

**Observations** (bold = best per row):
- MTSIR3-GAN is competitive with both a strong non-GAN baseline (TimesNet) and a GAN baseline (SSGAN), and is best on PSM at the higher missing rate.
- No single method dominates across all datasets and missing rates.
- Training is stable thanks to the regularized relativistic loss.

> A follow-up empirical study examining when adversarial refinement does and does not help lives in [`FMGAN/`](FMGAN/).

## 🔧 Hyperparameter Tuning

Key hyperparameters and recommended ranges:

| Parameter          | Description                      | Recommended Range  |
|--------------------|----------------------------------|--------------------|
| `batch_size`       | Training batch size              | 32-128             |
| `WidthPerStage`    | Network width (channels)         | [512, 512, 512, 512] - [1024, 1024, 1024, 1024] |
| `BlocksPerStage`   | Residual blocks per stage        | [2, 2, 2, 2]       |
| `learning_rate`    | Initial LR (with cosine decay)   | 1e-4 to 5e-5       |
| `gamma`            | R₁/R₂ regularization strength    | 0.05 - 1.0         |
| `NoiseDimension`   | Latent noise dimension           | 64-256             |

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete training instructions.

## 📚 Documentation

- **[README_CN.md](README_CN.md)** - 中文文档
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete usage instructions for all models
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure and file organization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{mtsir3gan2025,
  title={MTSIR3-GAN: Adapting R3GAN for Robust Multivariate Time Series Imputation},
  author={He, Yufeng},
  year={2025}
}
```

## 📚 References

Key papers and resources:

- **R3GAN**: Huang, Gokaslan, Kuleshov, Tompkin. "The GAN is dead; long live the GAN! A Modern GAN Baseline" (NeurIPS 2024)
- **SSGAN**: Miao et al. "Generative Semi-supervised Learning for Multivariate Time Series Imputation" (AAAI 2021)
- **TimesNet**: Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
- **BRITS**: Cao et al. "BRITS: Bidirectional Recurrent Imputation for Time Series" (NeurIPS 2018)

## 📧 Contact

- **Author**: Yufeng He
- **GitHub**: [@he-yufeng](https://github.com/he-yufeng)
- **Project Link**: https://github.com/he-yufeng/MTSIR3-GAN

## 📦 Third-Party Components & Licenses

This repository bundles code derived from external projects. The root MIT license
covers **only the original contributions** (the 1D / time-series adaptation, the
FMGAN study, GUI glue, and tooling). Bundled third-party code retains its own
license — see each folder's README:

| Folder | Upstream | License |
|--------|----------|---------|
| [`R3GAN/`](R3GAN/) | NVIDIA StyleGAN3 + R3GAN (NeurIPS 2024) | **NVIDIA Source Code License** (research / non-commercial) |
| [`TimesNet/`](TimesNet/) | THUML Time-Series-Library | MIT |
| [`SSGAN/`](SSGAN/) | Miao et al., AAAI 2021 | per original authors |

> ⚠️ Because `R3GAN/` includes NVIDIA-licensed code, that directory is **not**
> usable under MIT — respect the upstream non-commercial terms for any reuse.

## 📄 License

Original code in this repository is licensed under the MIT License — see
[LICENSE](LICENSE). See **Third-Party Components** above for bundled code under
other licenses.

## 🙏 Acknowledgments

- `R3GAN/` builds on [NVIDIA StyleGAN3](https://github.com/NVlabs/stylegan3) and [R3GAN](https://github.com/brownvc/R3GAN)
- `TimesNet/` is vendored from [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT)
- `SSGAN/` reproduces the AAAI 2021 baseline by Miao et al.
- GUI built with [Plotly Dash](https://dash.plotly.com/)
- Datasets from PhysioNet, the UCI ML Repository, and eBay
