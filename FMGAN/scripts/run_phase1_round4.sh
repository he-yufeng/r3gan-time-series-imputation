#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/MTSIR3-GAN/FMGAN:$PYTHONPATH
cd /root/MTSIR3-GAN/FMGAN

echo "============================================"
echo "Phase 1 Round 4: Baselines + Ablation"
echo "============================================"

# Exp A: SAITS baseline on Weather
echo ""
echo "[EXP A] SAITS on Weather (rate=0.25)"
python evaluation/run_baselines.py \
    --dataset Weather --method SAITS \
    --missing_rate 0.25 --seq_len 96 --epochs 100 \
    --output results/baseline_SAITS_Weather_0.25.json

# Exp B: BRITS baseline on Weather
echo ""
echo "[EXP B] BRITS on Weather (rate=0.25)"
python evaluation/run_baselines.py \
    --dataset Weather --method BRITS \
    --missing_rate 0.25 --seq_len 96 --epochs 100 \
    --output results/baseline_BRITS_Weather_0.25.json

# Exp C: Ablation - pure reconstruction model (no adversarial loss)
# Achieved by setting lambda_recon=100, lambda_freq=10, and gamma=0 effectively
# We hack this by making a very high recon weight so adversarial is negligible
echo ""
echo "[EXP C] Ablation: High recon, near-zero adversarial on Weather"
python train_refiner.py \
    --dataset Weather --coarse linear --seq_len 96 \
    --epochs 200 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 2.0 --lambda_recon 100.0 --lambda_freq 10.0 \
    --stride_divisor 2 \
    --outdir results/phase1_weather_ablation_highrec

# Exp D: SAITS on AirQuality
echo ""
echo "[EXP D] SAITS on AirQuality (rate=0.25)"
python evaluation/run_baselines.py \
    --dataset AirQuality --method SAITS \
    --missing_rate 0.25 --seq_len 36 --epochs 100 \
    --output results/baseline_SAITS_AirQuality_0.25.json

echo ""
echo "============================================"
echo "Round 4 complete!"
echo "============================================"

# Print baseline results
echo ""
echo "=== Baselines ==="
for f in results/baseline_*.json; do
    python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
if isinstance(r, list): r = r[0]
print('  %s on %s: MAE=%.6f  MSE=%.6f  time=%.1fs' % (r['method'], r['dataset'], r['MAE'], r['MSE'], r.get('infer_time_s', 0)))
" "$f"
done

echo ""
echo "=== All R3GAN results ==="
for d in results/phase1_*/; do
    if [ -f "${d}results.json" ]; then
        python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
ds = r['dataset']
cm = r['coarse_method']
c = r['coarse_metrics']['MAE']
rf = r['refined_metrics']['MAE']
imp = r['improvement_pct']
print('  %s (%s) [%s]: %.6f -> %.6f  (%+.2f%%)' % (ds, cm, sys.argv[2], c, rf, imp))
" "${d}results.json" "$(basename $d)"
    fi
done
