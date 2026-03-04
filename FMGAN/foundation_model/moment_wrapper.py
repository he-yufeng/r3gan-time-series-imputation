"""
MOMENT Foundation Model Wrapper for Time Series Imputation.

MOMENT (ICML 2024) is a family of open time-series foundation models.
We use it as the "coarse imputation" stage in our Coarse-to-Fine pipeline.

Compatibility notes (momentfm 0.1.4):
- Task name must be "reconstruction" (not "imputation")
- Forward uses keyword-only args: model.reconstruct(x_enc=..., input_mask=...)
- Output attribute is .reconstruction (not .output)
- In China, set HF_ENDPOINT=https://hf-mirror.com before use
"""

import os
import torch
import torch.nn as nn
import numpy as np


class MOMENTImputer(nn.Module):
    """
    Wraps MOMENT for time series imputation via reconstruction.

    Usage:
        imputer = MOMENTImputer(frozen=True)
        X_coarse = imputer(X_observed, mask)
    """

    def __init__(self, model_name='AutonLab/MOMENT-1-large', frozen=True, device='cuda'):
        super().__init__()

        # Set HF mirror for China if not already set
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        import json
        import argparse
        from huggingface_hub import hf_hub_download
        from momentfm import MOMENTPipeline

        # Download config and weights
        config_file = hf_hub_download(repo_id=model_name, filename='config.json')
        weights_file = hf_hub_download(repo_id=model_name, filename='pytorch_model.bin')

        with open(config_file) as f:
            config_dict = json.load(f)

        # Use "reconstruction" task (equivalent to imputation in v0.1.4)
        config_dict['task_name'] = 'reconstruction'
        config = argparse.Namespace(**config_dict)

        self.model = MOMENTPipeline(config=config)
        state_dict = torch.load(weights_file, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.init()

        self.frozen = frozen
        self._device = device
        self.seq_len = config_dict.get('seq_len', 512)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, X_observed, mask):
        """
        Args:
            X_observed: (batch, seq_len, n_features) - observed values (missing=0)
            mask: (batch, seq_len, n_features) - binary mask (1=observed, 0=missing)

        Returns:
            X_coarse: (batch, seq_len, n_features) - coarse imputation result
        """
        batch_size, seq_len, n_features = X_observed.shape

        # MOMENT processes each feature (channel) independently
        # Reshape: (B, T, F) -> (B*F, 1, T)
        X_input = X_observed.permute(0, 2, 1).reshape(-1, 1, seq_len)
        mask_input = mask.permute(0, 2, 1).reshape(-1, seq_len)

        # Pad or truncate to MOMENT's expected input length
        moment_len = self.seq_len
        if seq_len < moment_len:
            pad_len = moment_len - seq_len
            X_padded = torch.nn.functional.pad(X_input, (0, pad_len), value=0)
            mask_padded = torch.nn.functional.pad(mask_input, (0, pad_len), value=0)
        elif seq_len > moment_len:
            X_padded = X_input[:, :, :moment_len]
            mask_padded = mask_input[:, :moment_len]
        else:
            X_padded = X_input
            mask_padded = mask_input

        # Run MOMENT reconstruction in small chunks to avoid NaN from numerical instability
        # MOMENT can produce NaN when processing too many sequences at once
        max_internal_batch = 128
        total = X_padded.shape[0]  # B*F
        imputed_chunks = []

        with torch.set_grad_enabled(not self.frozen):
            for start in range(0, total, max_internal_batch):
                end = min(start + max_internal_batch, total)
                chunk_out = self.model.reconstruct(
                    x_enc=X_padded[start:end],
                    input_mask=mask_padded[start:end],
                )
                chunk = chunk_out.reconstruction
                if chunk.ndim == 3:
                    chunk = chunk.squeeze(1)
                imputed_chunks.append(chunk)

        # Concatenate: (B*F, moment_len)
        X_imputed = torch.cat(imputed_chunks, dim=0)

        # Trim back to original length
        X_imputed = X_imputed[:, :seq_len]

        # Replace NaN with 0 (happens when a feature has no observations at all)
        X_imputed = torch.nan_to_num(X_imputed, nan=0.0)

        # Reshape back: (B*F, T) -> (B, T, F)
        X_coarse = X_imputed.reshape(batch_size, n_features, seq_len).permute(0, 2, 1)

        # Keep original observed values, only impute missing positions
        X_coarse = X_observed * mask + X_coarse * (1 - mask)

        return X_coarse

    def impute_numpy(self, X_observed_np, mask_np, batch_size=32):
        """Convenience method for numpy arrays."""
        device = self._device
        X_t = torch.from_numpy(X_observed_np).float().to(device)
        mask_t = torch.from_numpy(mask_np).float().to(device)

        results = []
        for i in range(0, len(X_t), batch_size):
            X_batch = X_t[i:i + batch_size]
            mask_batch = mask_t[i:i + batch_size]
            with torch.no_grad():
                result = self.forward(X_batch, mask_batch)
            results.append(result.cpu().numpy())

        return np.concatenate(results, axis=0)
