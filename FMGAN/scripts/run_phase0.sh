#!/bin/bash
# FMGAN Phase 0: Quick Validation (1-2 weeks on cloud GPU)
# Run this after setup_cloud_gpu.sh
#
# This script runs the full Go/No-Go validation:
# 1. Prepare datasets
# 2. Run SOTA baselines (SAITS, CSDI, BRITS)
# 3. Run MOMENT foundation model
# 4. Run MOMENT + GAN combination test
#
# Usage: bash scripts/run_phase0.sh

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

echo "=========================================="
echo "FMGAN Phase 0: Quick Validation"
echo "Project dir: $PROJ_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "=========================================="

mkdir -p results

# Step 1: Prepare datasets
echo ""
echo "[Step 1/4] Preparing datasets..."
python data/prepare_datasets.py

# Step 2: Run baselines on AirQuality (quickest dataset)
echo ""
echo "[Step 2/4] Running baselines on AirQuality..."
for METHOD in SAITS CSDI BRITS; do
    for RATE in 0.125 0.25 0.5; do
        echo "  >> $METHOD, rate=$RATE"
        python evaluation/run_baselines.py \
            --dataset AirQuality \
            --method $METHOD \
            --missing_rate $RATE \
            --seq_len 36 \
            --epochs 100 \
            --output "results/baseline_${METHOD}_AirQuality_${RATE}.json"
    done
done

# Step 3: Run MOMENT baseline
echo ""
echo "[Step 3/4] Running MOMENT baseline..."
for RATE in 0.125 0.25 0.5; do
    echo "  >> MOMENT, rate=$RATE"
    python evaluation/run_baselines.py \
        --dataset AirQuality \
        --method MOMENT \
        --missing_rate $RATE \
        --seq_len 36 \
        --output "results/baseline_MOMENT_AirQuality_${RATE}.json"
done

# Step 4: Run combination test (THE KEY EXPERIMENT)
echo ""
echo "[Step 4/4] Running MOMENT + GAN combination test..."
for RATE in 0.125 0.25 0.5; do
    echo "  >> Combination test, rate=$RATE"
    python evaluation/run_combination_test.py \
        --dataset AirQuality \
        --missing_rate $RATE \
        --seq_len 36 \
        --output "results/combination_AirQuality_${RATE}.json"
done

# Summary
echo ""
echo "=========================================="
echo "Phase 0 Complete! Check results/ directory."
echo ""
echo "Go/No-Go Decision:"
echo "  - Compare results/combination_*.json"
echo "  - If MOMENT+GAN > MOMENT alone: PROCEED to Phase 1"
echo "  - If MOMENT+GAN ≈ MOMENT: try adding freq loss"
echo "  - If MOMENT+GAN < MOMENT: rethink approach"
echo "=========================================="
