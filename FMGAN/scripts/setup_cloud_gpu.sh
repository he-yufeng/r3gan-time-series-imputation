#!/bin/bash
# FMGAN Cloud GPU Setup Script (AutoDL / similar platforms)
#
# Usage:
#   source scripts/setup_cloud_gpu.sh
#   (use 'source' not 'bash' so conda activate works in current shell)

set -e

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "FMGAN Environment Setup"
echo "Project dir: $PROJ_DIR"
echo "=========================================="

# 0. Enable conda in script
eval "$(conda shell.bash hook)" 2>/dev/null || true

# 1. Create conda environment
echo "[1/6] Creating conda environment..."
conda create -n fmgan python=3.10 -y
conda activate fmgan

# 2. Install PyTorch (CUDA 11.8)
# AutoDL images typically have CUDA pre-installed; adjust cu118/cu121 as needed
echo "[2/6] Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
echo "[3/6] Installing project dependencies..."
cd "$PROJ_DIR"
pip install -r requirements.txt

# 4. Set PYTHONPATH so all FMGAN modules can be imported
echo "[4/6] Setting PYTHONPATH..."
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"
# Also persist for future shells
echo "export PYTHONPATH=\"$PROJ_DIR:\$PYTHONPATH\"" >> ~/.bashrc

# 5. Download MOMENT pretrained weights (cache locally to avoid re-download)
echo "[5/6] Downloading MOMENT pretrained weights..."
python -c "
from momentfm import MOMENTPipeline
model = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large', model_kwargs={'task_name': 'imputation'})
print('MOMENT downloaded successfully.')
"

# 6. Prepare datasets (download + convert to .npz)
echo "[6/6] Preparing datasets..."
python data/prepare_datasets.py

echo "=========================================="
echo "Setup complete!"
echo "  conda activate fmgan"
echo "  export PYTHONPATH=$PROJ_DIR:\$PYTHONPATH"
echo "  cd $PROJ_DIR && bash scripts/run_phase0.sh"
echo "=========================================="
