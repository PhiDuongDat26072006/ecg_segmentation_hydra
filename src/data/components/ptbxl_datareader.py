"""
PTB-XL data reader for classification fine-tuning.

Reads PTB-XL dataset and extracts binary labels:
  - Label 1: AFIB or AFLT (Atrial Fibrillation / Atrial Flutter)
  - Label 0: Everything else

Uses strat_fold column for train/val/test split:
  - Folds 1-8: Train
  - Fold 9: Validation
  - Fold 10: Test
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
import wfdb


# PTB-XL lead name to index mapping (standard 12-lead order)
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _is_afib_or_aflt(scp_codes_dict):
    """Check if scp_codes contain AFIB or AFLT.

    Args:
        scp_codes_dict: dict of SCP codes with likelihoods, e.g. {'AFIB': 100.0}

    Returns:
        1 if AFIB or AFLT is present, 0 otherwise
    """
    for code in scp_codes_dict.keys():
        if code.upper() in ('AFIB', 'AFLT'):
            return 1
    return 0


def load_ptbxl_cls_tensors(data_dir, leads=None, sampling_rate=500):
    """Load PTB-XL dataset for classification.

    Args:
        data_dir: path to the PTB-XL root directory (containing ptbxl_database.csv)
        leads: list of lead names to use, e.g. ['I', 'II']. If None, uses ['I', 'II'].
        sampling_rate: 500 or 100 Hz

    Returns:
        dict with keys 'train', 'val', 'test', each containing:
            X: torch.Tensor (N * n_leads, 1, signal_length) - signal data
            y_cls: torch.Tensor (N * n_leads,) - classification labels (int64)
    """
    if leads is None:
        leads = ['I', 'II']

    # Normalize lead names to uppercase
    leads = [l.upper() for l in leads]
    lead_indices = [LEAD_NAMES.index(l) for l in leads]

    # Read metadata
    csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy ptbxl_database.csv tại: {csv_path}")

    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Parse scp_codes from string to dict
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    # Extract binary labels
    df['cls_label'] = df['scp_codes'].apply(_is_afib_or_aflt)

    # Select signal directory
    if sampling_rate == 500:
        signal_dir = 'records500'
        signal_length = 5000
    else:
        signal_dir = 'records100'
        signal_length = 1000

    # Split by strat_fold
    splits = {
        'train': df[df['strat_fold'].isin(range(1, 9))],
        'val': df[df['strat_fold'] == 9],
        'test': df[df['strat_fold'] == 10],
    }

    result = {}
    for split_name, split_df in splits.items():
        all_waves = []
        all_labels = []

        for _, row in split_df.iterrows():
            # Build path to signal file
            filename = row['filename_hr'] if sampling_rate == 500 else row['filename_lr']
            record_path = os.path.join(data_dir, filename)

            # Load signal
            try:
                signal, _ = wfdb.rdsamp(record_path)
            except Exception as e:
                print(f"Warning: Không đọc được {record_path}: {e}")
                continue

            # signal shape: (signal_length, 12)
            label = row['cls_label']

            # Extract selected leads
            for lead_idx in lead_indices:
                wave = signal[:, lead_idx]  # (signal_length,)

                # Pad or truncate to expected length
                if len(wave) < signal_length:
                    wave = np.pad(wave, (0, signal_length - len(wave)), mode='constant')
                elif len(wave) > signal_length:
                    wave = wave[:signal_length]

                all_waves.append(wave)
                all_labels.append(label)

        # Convert to tensors
        all_waves = np.array(all_waves, dtype=np.float32)       # (N * n_leads, signal_length)
        all_labels = np.array(all_labels, dtype=np.int64)         # (N * n_leads,)

        X_torch = torch.tensor(all_waves).unsqueeze(1)            # (N * n_leads, 1, signal_length)
        y_cls_torch = torch.tensor(all_labels)                     # (N * n_leads,)

        result[split_name] = {
            'X': X_torch,
            'y_cls': y_cls_torch,
        }

        n_pos = int(all_labels.sum())
        n_total = len(all_labels)
        print(f"  PTB-XL [{split_name}]: {n_total} signals "
              f"({n_pos} AFIB/AFLT, {n_total - n_pos} other)")

    return result
