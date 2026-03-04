"""
Phase 0.2: Run SOTA baselines using PyPOTS unified framework.

Usage:
    python evaluation/run_baselines.py --dataset AirQuality --missing_rate 0.25
    python evaluation/run_baselines.py --all  # Run all configurations
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Ensure FMGAN root is on the path regardless of working directory
_FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FMGAN_ROOT not in sys.path:
    sys.path.insert(0, _FMGAN_ROOT)

from evaluation.metrics import compute_all_metrics


def run_pypots_baseline(method_name, dataset_name, missing_rate=0.25,
                        missing_pattern='point', seq_len=96, seed=42, epochs=100):
    """
    Run a single PyPOTS baseline method.

    Returns:
        dict with metrics and timing info
    """
    from pypots.imputation import SAITS, CSDI, BRITS
    import tsdb

    print(f"\n{'='*60}")
    print(f"Running {method_name} on {dataset_name} (rate={missing_rate}, pattern={missing_pattern})")
    print(f"{'='*60}")

    # Load dataset via PyPOTS ecosystem
    pypots_map = {
        'PhysioNet2012': 'physionet_2012',
        'ETTh1': 'ETTh1',
        'ETTh2': 'ETTh2',
    }

    if dataset_name in pypots_map:
        data = tsdb.load(pypots_map[dataset_name])
        X = data['X'] if isinstance(data, dict) else data
    else:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'datasets', dataset_name, 'data.npz'
        )
        loaded = np.load(data_path)
        X_full = loaded['X']
        # Window into samples
        samples = []
        stride = seq_len // 2
        for start in range(0, len(X_full) - seq_len + 1, stride):
            samples.append(X_full[start:start + seq_len])
        X = np.stack(samples)

    if X.ndim == 2:
        # Reshape flat data into windows
        n_features = X.shape[-1]
        samples = []
        stride = seq_len // 2
        for start in range(0, len(X) - seq_len + 1, stride):
            samples.append(X[start:start + seq_len])
        X = np.stack(samples)

    # Normalize
    from sklearn.preprocessing import StandardScaler
    orig_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(orig_shape).astype(np.float32)

    # Split: 70% train, 15% val, 15% test
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]

    # Apply artificial missing
    rng = np.random.default_rng(seed)
    X_test_intact = X_test.copy()

    if missing_pattern == 'point':
        test_mask = (rng.random(X_test.shape) > missing_rate).astype(np.float32)
    else:
        from data.unified_loader import MissingPatternGenerator  # noqa: E402
        gen = MissingPatternGenerator()
        if missing_pattern == 'subsequence':
            test_mask = gen.subsequence_missing(X_test.shape, missing_rate, rng)
        else:
            test_mask = gen.block_missing(X_test.shape, missing_rate, rng)

    indicating_mask = (1 - test_mask).astype(np.float32)
    X_test_missing = X_test.copy()
    X_test_missing[test_mask == 0] = np.nan

    # Also create training data with missing values
    train_mask = (rng.random(X_train.shape) > missing_rate).astype(np.float32)
    X_train_missing = X_train.copy()
    X_train_missing[train_mask == 0] = np.nan

    val_mask = (rng.random(X_val.shape) > missing_rate).astype(np.float32)
    X_val_missing = X_val.copy()
    X_val_missing[val_mask == 0] = np.nan

    # Prepare PyPOTS dataset dict
    dataset_dict = {
        "n_steps": X.shape[1],
        "n_features": X.shape[2],
        "train_X": X_train_missing,
        "val_X": X_val_missing,
        "test_X": X_test_missing,
        "test_X_intact": X_test_intact,
        "test_X_indicating_mask": indicating_mask,
    }

    # Select and configure model
    n_features = X.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'SAITS': lambda: SAITS(
            n_steps=dataset_dict['n_steps'],
            n_features=n_features,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            epochs=epochs,
            batch_size=32,
            device=device,
        ),
        'CSDI': lambda: CSDI(
            n_steps=dataset_dict['n_steps'],
            n_features=n_features,
            n_layers=4,
            n_heads=8,
            n_channels=64,
            d_time_embedding=128,
            d_feature_embedding=16,
            d_diffusion_embedding=128,
            n_diffusion_steps=50,
            epochs=epochs,
            batch_size=16,
            device=device,
        ),
        'BRITS': lambda: BRITS(
            n_steps=dataset_dict['n_steps'],
            n_features=n_features,
            rnn_hidden_size=128,
            epochs=epochs,
            batch_size=32,
            device=device,
        ),
    }

    if method_name not in model_configs:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(model_configs.keys())}")

    model = model_configs[method_name]()

    # Train
    print(f"Training {method_name}...")
    t_start = time.time()
    model.fit(
        train_set={'X': X_train_missing},
        val_set={'X': X_val_missing, 'X_ori': X_val},
    )
    train_time = time.time() - t_start

    # Evaluate
    print(f"Evaluating {method_name}...")
    t_start = time.time()
    results = model.predict(test_set={'X': X_test_missing})
    infer_time = time.time() - t_start

    X_pred = results['imputation']

    # Compute metrics on indicating_mask positions only
    metrics = compute_all_metrics(X_pred, X_test_intact, indicating_mask)
    metrics['train_time_s'] = round(train_time, 1)
    metrics['infer_time_s'] = round(infer_time, 3)
    metrics['method'] = method_name
    metrics['dataset'] = dataset_name
    metrics['missing_rate'] = missing_rate
    metrics['missing_pattern'] = missing_pattern

    print(f"\nResults for {method_name} on {dataset_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return metrics


def run_moment_baseline(dataset_name, missing_rate=0.25, missing_pattern='point',
                        seq_len=96, seed=42):
    """Run MOMENT foundation model as a baseline."""
    from foundation_model.moment_wrapper import MOMENTImputer
    from data.unified_loader import TimeSeriesImputationDataset

    print(f"\n{'='*60}")
    print(f"Running MOMENT on {dataset_name} (rate={missing_rate}, pattern={missing_pattern})")
    print(f"{'='*60}")

    # Load test data using unified loader
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'datasets', dataset_name, 'data.npz'
    )
    loaded = np.load(data_path)
    X_full = loaded['X']
    n = len(X_full)
    X_test_raw = X_full[int(0.85 * n):]

    dataset = TimeSeriesImputationDataset(
        X_test_raw, seq_len=seq_len, missing_rate=missing_rate,
        missing_pattern=missing_pattern, seed=seed,
    )

    # Collect all test data
    X_intact_all, X_obs_all, mask_all, indicating_all = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        X_intact_all.append(item['X_intact'].numpy())
        X_obs_all.append(item['X_observed'].numpy())
        mask_all.append(item['mask'].numpy())
        indicating_all.append(item['indicating_mask'].numpy())

    X_intact_np = np.stack(X_intact_all)
    X_obs_np = np.stack(X_obs_all)
    mask_np = np.stack(mask_all)
    indicating_np = np.stack(indicating_all)

    # Run MOMENT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imputer = MOMENTImputer(frozen=True, device=device)
    imputer.to(device)

    t_start = time.time()
    X_pred = imputer.impute_numpy(X_obs_np, mask_np, batch_size=32)
    infer_time = time.time() - t_start

    metrics = compute_all_metrics(X_pred, X_intact_np, indicating_np)
    metrics['infer_time_s'] = round(infer_time, 3)
    metrics['method'] = 'MOMENT'
    metrics['dataset'] = dataset_name
    metrics['missing_rate'] = missing_rate
    metrics['missing_pattern'] = missing_pattern

    print(f"\nResults for MOMENT on {dataset_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run FMGAN Phase 0 baseline evaluation')
    parser.add_argument('--dataset', type=str, default='AirQuality',
                        help='Dataset name')
    parser.add_argument('--method', type=str, default='all',
                        help='Method name or "all"')
    parser.add_argument('--missing_rate', type=float, default=0.25)
    parser.add_argument('--missing_pattern', type=str, default='point')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/phase0_baselines.json')
    parser.add_argument('--all', action='store_true', help='Run all Phase 0 experiments')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results = []

    if args.all:
        # Phase 0 quick validation: 2 datasets × 3 missing rates × core methods
        datasets = ['AirQuality', 'ETTh1']
        missing_rates = [0.125, 0.25, 0.5]
        methods = ['SAITS', 'CSDI', 'BRITS']

        for ds in datasets:
            for rate in missing_rates:
                for method in methods:
                    try:
                        result = run_pypots_baseline(
                            method, ds, rate, args.missing_pattern,
                            args.seq_len, args.seed, args.epochs
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"ERROR: {method} on {ds} (rate={rate}): {e}")

                # Also run MOMENT
                try:
                    result = run_moment_baseline(ds, rate, args.missing_pattern,
                                                args.seq_len, args.seed)
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: MOMENT on {ds} (rate={rate}): {e}")
    else:
        methods = ['SAITS', 'CSDI', 'BRITS'] if args.method == 'all' else [args.method]
        for method in methods:
            if method == 'MOMENT':
                result = run_moment_baseline(args.dataset, args.missing_rate,
                                            args.missing_pattern, args.seq_len, args.seed)
            else:
                result = run_pypots_baseline(
                    method, args.dataset, args.missing_rate, args.missing_pattern,
                    args.seq_len, args.seed, args.epochs
                )
            all_results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<15} {'Dataset':<15} {'Rate':<8} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'Time(s)':<10}")
    print(f"{'='*80}")
    for r in all_results:
        print(f"{r['method']:<15} {r['dataset']:<15} {r['missing_rate']:<8.3f} "
              f"{r['MAE']:<10.6f} {r['MSE']:<10.6f} {r['RMSE']:<10.6f} "
              f"{r.get('infer_time_s', 'N/A'):<10}")


if __name__ == '__main__':
    main()
