#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/MTSIR3-GAN/FMGAN:$PYTHONPATH
cd /root/MTSIR3-GAN/FMGAN

echo "============================================"
echo "Phase 1 Round 3: R3GAN as standalone imputer"
echo "============================================"

# Exp A: Weather, zero coarse (R3GAN alone)
echo ""
echo "[EXP A] Weather - R3GAN standalone (coarse=zero)"
python train_refiner.py \
    --dataset Weather --coarse zero --seq_len 96 \
    --epochs 300 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --stride_divisor 2 \
    --outdir results/phase1_weather_zero

# Exp B: Electricity, zero coarse
echo ""
echo "[EXP B] Electricity - R3GAN standalone (coarse=zero)"
python train_refiner.py \
    --dataset Electricity --coarse zero --seq_len 96 \
    --epochs 100 --batch_size 64 \
    --width 128 --n_stages 3 --n_blocks 2 --cardinality 32 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --stride_divisor 4 \
    --outdir results/phase1_elec_zero

# Exp C: AirQuality, zero coarse
echo ""
echo "[EXP C] AirQuality - R3GAN standalone (coarse=zero)"
python train_refiner.py \
    --dataset AirQuality --coarse zero --seq_len 36 \
    --epochs 300 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --stride_divisor 4 \
    --outdir results/phase1_air_zero

echo ""
echo "============================================"
echo "Round 3 complete!"
echo "============================================"
for d in results/phase1_*/; do
    if [ -f "${d}results.json" ]; then
        echo "--- $(basename $d) ---"
        python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
ds = r['dataset']
cm = r['coarse_method']
c = r['coarse_metrics']['MAE']
rf = r['refined_metrics']['MAE']
imp = r['improvement_pct']
print('  %s (%s): %.6f -> %.6f  (%+.2f%%)' % (ds, cm, c, rf, imp))
" "${d}results.json"
    fi
done
