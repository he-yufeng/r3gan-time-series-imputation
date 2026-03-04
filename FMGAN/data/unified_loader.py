"""
Unified data loader for FMGAN experiments.
Supports multiple datasets, missing patterns, and missing rates.
Uses PyGrinder for standardized missing value generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MissingPatternGenerator:
    """Generate missing value masks with different patterns."""

    @staticmethod
    def point_missing(shape, rate, rng=None):
        """MCAR (Missing Completely At Random) - random point missing."""
        rng = rng or np.random.default_rng()
        mask = rng.random(shape) > rate  # True = observed
        return mask.astype(np.float32)

    @staticmethod
    def subsequence_missing(shape, rate, min_len=5, max_len=20, rng=None):
        """Missing subsequences in individual features."""
        rng = rng or np.random.default_rng()
        n_samples, seq_len, n_features = shape
        mask = np.ones(shape, dtype=np.float32)

        for i in range(n_samples):
            for j in range(n_features):
                total_missing = 0
                target_missing = int(seq_len * rate)
                while total_missing < target_missing:
                    start = rng.integers(0, seq_len)
                    length = rng.integers(min_len, min(max_len, seq_len - start) + 1)
                    mask[i, start:start + length, j] = 0.0
                    total_missing += length
        return mask

    @staticmethod
    def block_missing(shape, rate, rng=None):
        """Block missing - rectangular regions missing across multiple features."""
        rng = rng or np.random.default_rng()
        n_samples, seq_len, n_features = shape
        mask = np.ones(shape, dtype=np.float32)

        for i in range(n_samples):
            total_elements = seq_len * n_features
            target_missing = int(total_elements * rate)
            current_missing = 0

            while current_missing < target_missing:
                t_start = rng.integers(0, seq_len)
                t_len = rng.integers(1, max(2, seq_len // 5))
                f_start = rng.integers(0, n_features)
                f_len = rng.integers(1, max(2, n_features // 3))

                t_end = min(t_start + t_len, seq_len)
                f_end = min(f_start + f_len, n_features)
                mask[i, t_start:t_end, f_start:f_end] = 0.0
                current_missing += (t_end - t_start) * (f_end - f_start)

        return mask


class TimeSeriesImputationDataset(Dataset):
    """
    Unified dataset for time series imputation.

    Returns:
        X_intact: ground truth (complete data)
        X_observed: observed data (with missing values zeroed out)
        mask: binary mask (1 = observed, 0 = missing)
        indicating_mask: mask for evaluation (only artificially masked positions)
    """

    def __init__(self, X, seq_len=96, missing_rate=0.25, missing_pattern='point',
                 stride=None, seed=42):
        """
        Args:
            X: numpy array of shape (total_len, n_features) - the full time series
            seq_len: length of each sample window
            missing_rate: fraction of values to mask
            missing_pattern: 'point', 'subsequence', or 'block'
            stride: step size between windows (default: seq_len // 2)
            seed: random seed for reproducibility
        """
        self.seq_len = seq_len
        self.missing_rate = missing_rate
        self.missing_pattern = missing_pattern
        self.rng = np.random.default_rng(seed)

        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sliding windows
        stride = stride or seq_len // 2
        self.samples = []
        for start in range(0, len(X_scaled) - seq_len + 1, stride):
            window = X_scaled[start:start + seq_len]
            if not np.isnan(window).all():
                self.samples.append(window.astype(np.float32))

        self.samples = np.stack(self.samples)  # (N, seq_len, n_features)

        # Handle originally missing values (NaN)
        self.original_mask = (~np.isnan(self.samples)).astype(np.float32)
        self.samples = np.nan_to_num(self.samples, nan=0.0)

        # Generate artificial missing masks
        gen = MissingPatternGenerator()
        gen_func = {
            'point': gen.point_missing,
            'subsequence': gen.subsequence_missing,
            'block': gen.block_missing,
        }[missing_pattern]

        self.artificial_mask = gen_func(self.samples.shape, missing_rate, self.rng)
        # Combined mask: observed only if both originally present AND not artificially masked
        self.combined_mask = self.original_mask * self.artificial_mask
        # Indicating mask: positions that are artificially masked (for evaluation)
        self.indicating_mask = self.original_mask * (1 - self.artificial_mask)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_intact = torch.from_numpy(self.samples[idx])       # (seq_len, features)
        mask = torch.from_numpy(self.combined_mask[idx])      # (seq_len, features)
        indicating = torch.from_numpy(self.indicating_mask[idx])
        X_observed = X_intact * mask                          # Zero out missing

        return {
            'X_intact': X_intact,
            'X_observed': X_observed,
            'mask': mask,
            'indicating_mask': indicating,
        }


def load_dataset(name, seq_len=96, missing_rate=0.25, missing_pattern='point',
                 batch_size=64, seed=42, split='train', num_workers=4):
    """
    Load a dataset by name and return train/val/test DataLoaders.

    Args:
        name: dataset name (AirQuality, ETTh1, ETTh2, PhysioNet2012, etc.)
        seq_len: sample window length
        missing_rate: fraction of values to artificially mask
        missing_pattern: 'point', 'subsequence', or 'block'
        batch_size: batch size
        seed: random seed
        split: 'train', 'val', or 'test'
        num_workers: dataloader workers

    Returns:
        DataLoader for the specified split
    """
    import pypots.data
    import pygrinder

    # Use PyPOTS unified loading where available
    pypots_datasets = {
        'PhysioNet2012': 'physionet_2012',
        'ETTh1': 'ETTh1',
        'ETTh2': 'ETTh2',
    }

    if name in pypots_datasets:
        data = pypots.data.load_specific_dataset(pypots_datasets[name])
        X = data['train_X'] if split == 'train' else data.get('test_X', data['train_X'])
    else:
        import os
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', name)
        npz_path = os.path.join(data_dir, 'data.npz')
        if os.path.exists(npz_path):
            loaded = np.load(npz_path)
            X_full = loaded['X']
            # Simple 70/15/15 split
            n = len(X_full)
            if split == 'train':
                X = X_full[:int(0.7 * n)]
            elif split == 'val':
                X = X_full[int(0.7 * n):int(0.85 * n)]
            else:
                X = X_full[int(0.85 * n):]
        else:
            raise FileNotFoundError(f"Dataset {name} not found at {data_dir}")

    # If X is 3D (N, T, F), flatten to 2D (N*T, F) for windowing
    if X.ndim == 3:
        n_samples, t, f = X.shape
        X = X.reshape(-1, f)

    dataset = TimeSeriesImputationDataset(
        X, seq_len=seq_len, missing_rate=missing_rate,
        missing_pattern=missing_pattern, seed=seed,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True, drop_last=(split == 'train'),
    )
    return loader
