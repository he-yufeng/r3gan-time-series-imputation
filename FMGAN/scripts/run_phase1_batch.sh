#!/bin/bash
# Phase 1: Batch experiments
# Run inside tmux on cloud GPU
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/MTSIR3-GAN/FMGAN:$PYTHONPATH
cd /root/MTSIR3-GAN/FMGAN

echo "============================================"
echo "Phase 1 Batch Experiments"
echo "============================================"

# Exp 1: AirQuality with augmentation + higher overlap (stride_divisor=4 → ~4x more samples)
echo ""
echo "[EXP 1/4] AirQuality + augment + stride/4"
python train_refiner.py \
    --dataset AirQuality --coarse linear --seq_len 36 \
    --epochs 300 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --augment --stride_divisor 4 \
    --outdir results/phase1_air_aug

# Exp 2: Weather with augmentation
echo ""
echo "[EXP 2/4] Weather + augment + stride/4"
python train_refiner.py \
    --dataset Weather --coarse linear --seq_len 96 \
    --epochs 200 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --augment --stride_divisor 4 \
    --outdir results/phase1_weather_aug

# Exp 3: Electricity (140K samples, 370 features — use larger model)
echo ""
echo "[EXP 3/4] Electricity (large dataset)"
python train_refiner.py \
    --dataset Electricity --coarse linear --seq_len 96 \
    --epochs 50 --batch_size 64 \
    --width 128 --n_stages 3 --n_blocks 2 --cardinality 32 \
    --lr 2e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --augment --stride_divisor 4 \
    --outdir results/phase1_electricity

# Exp 4: Weather with mean fill as coarse (test different coarse methods)
echo ""
echo "[EXP 4/4] Weather + mean fill coarse"
python train_refiner.py \
    --dataset Weather --coarse mean --seq_len 96 \
    --epochs 200 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --augment --stride_divisor 4 \
    --outdir results/phase1_weather_mean

echo ""
echo "============================================"
echo "All Phase 1 experiments complete!"
echo "============================================"

# Print summary
echo ""
for d in results/phase1_*/; do
    if [ -f "$d/results.json" ]; then
        echo "--- $d ---"
        python -c "import json; r=json.load(open('${d}results.json')); print(f\"  {r['dataset']} ({r['coarse_method']}): coarse MAE={r['coarse_metrics']['MAE']:.6f} → refined MAE={r['refined_metrics']['MAE']:.6f} ({r['improvement_pct']:+.2f}%)\")"
    fi
done
