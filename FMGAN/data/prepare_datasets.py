"""
Download and prepare all datasets for FMGAN evaluation.
Uses TSDB for standardized downloading, converts everything to .npz format.
"""

import os
import sys
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets')


def prepare_airquality():
    """Convert existing AirQuality .txt files to .npz format."""
    save_dir = os.path.join(DATASET_DIR, 'AirQuality')
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print("  AirQuality: data.npz already exists, skipping.")
        return

    ground_path = os.path.join(save_dir, 'pm25_ground.txt')
    if not os.path.exists(ground_path):
        print("  WARNING: AirQuality pm25_ground.txt not found!")
        return

    print("  Converting AirQuality .txt → .npz...")
    df = pd.read_csv(ground_path)
    # Drop datetime column, keep only numeric station data
    numeric_cols = [c for c in df.columns if c != 'datetime']
    X = df[numeric_cols].values.astype(np.float32)
    # Replace empty strings / non-numeric with NaN
    np.savez(npz_path, X=X)
    print(f"  AirQuality saved: shape={X.shape} ({X.shape[0]} timesteps × {X.shape[1]} stations)")


def prepare_psm():
    """Convert existing PSM .csv files to .npz format."""
    save_dir = os.path.join(DATASET_DIR, 'PSM')
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print("  PSM: data.npz already exists, skipping.")
        return

    # PSM has test.csv
    csv_path = os.path.join(save_dir, 'test.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: PSM test.csv not found!")
        return

    print("  Converting PSM .csv → .npz...")
    df = pd.read_csv(csv_path)
    X = df.values.astype(np.float32)
    np.savez(npz_path, X=X)
    print(f"  PSM saved: shape={X.shape}")


def prepare_physionet2012():
    """Download PhysioNet 2012 via TSDB."""
    save_dir = os.path.join(DATASET_DIR, 'PhysioNet2012')
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print("  PhysioNet2012: data.npz already exists, skipping.")
        return

    print("  Downloading PhysioNet2012 via TSDB...")
    try:
        import tsdb
        data = tsdb.load('physionet_2012')
        X = data['X'].astype(np.float32)
        np.savez(npz_path, X=X)
        print(f"  PhysioNet2012 saved: shape={X.shape}")
    except Exception as e:
        print(f"  ERROR downloading PhysioNet2012: {e}")
        print("  You can manually download from: https://physionet.org/content/challenge-2012/1.0.0/")


def prepare_ett(name):
    """Download ETT dataset via TSDB."""
    save_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print(f"  {name}: data.npz already exists, skipping.")
        return

    print(f"  Downloading {name} via TSDB...")
    try:
        import tsdb
        # TSDB uses lowercase names like 'electricity_transformer_temperature_h1'
        tsdb_name = name.lower().replace('ett', 'electricity_transformer_temperature_')
        data = tsdb.load(tsdb_name)
        X = data['X'].astype(np.float32)
        np.savez(npz_path, X=X)
        print(f"  {name} saved: shape={X.shape}")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")


def prepare_weather():
    """Prepare Weather dataset."""
    save_dir = os.path.join(DATASET_DIR, 'Weather')
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print("  Weather: data.npz already exists, skipping.")
        return

    # Check if weather.csv exists (from PURE-GUI uploaded files)
    gui_csv = os.path.join(DATASET_DIR, '..', 'PURE-GUIv2.0', 'uploaded_files', 'weather.csv')
    if os.path.exists(gui_csv):
        print("  Converting Weather .csv → .npz...")
        df = pd.read_csv(gui_csv)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values.astype(np.float32)
        np.savez(npz_path, X=X)
        print(f"  Weather saved: shape={X.shape}")
    else:
        print("  Weather: CSV not found. Will attempt TSDB download...")
        try:
            import tsdb
            data = tsdb.load('weather')
            X = data['X'].astype(np.float32)
            np.savez(npz_path, X=X)
            print(f"  Weather saved: shape={X.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Please manually place weather.csv in datasets/Weather/")


def prepare_electricity():
    """Download Electricity dataset via TSDB."""
    save_dir = os.path.join(DATASET_DIR, 'Electricity')
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, 'data.npz')
    if os.path.exists(npz_path):
        print("  Electricity: data.npz already exists, skipping.")
        return

    print("  Downloading Electricity via TSDB (this may take a while)...")
    try:
        import tsdb
        data = tsdb.load('electricity_load_diagrams')
        X = data['X'].astype(np.float32)
        np.savez(npz_path, X=X)
        print(f"  Electricity saved: shape={X.shape}")
    except Exception as e:
        print(f"  ERROR downloading Electricity: {e}")


def main():
    print("=" * 50)
    print("Preparing datasets for FMGAN")
    print(f"Dataset directory: {os.path.abspath(DATASET_DIR)}")
    print("=" * 50)

    # Priority 1: Convert existing local data (no download needed)
    print("\n[Local data conversion]")
    prepare_airquality()
    prepare_psm()

    # Priority 2: Download standard benchmarks
    print("\n[Downloading benchmarks via TSDB]")
    prepare_physionet2012()
    for name in ['ETTh1', 'ETTh2']:
        prepare_ett(name)
    prepare_weather()
    prepare_electricity()

    # Summary
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    for name in ['AirQuality', 'PSM', 'PhysioNet2012', 'ETTh1', 'ETTh2', 'Weather', 'Electricity']:
        npz = os.path.join(DATASET_DIR, name, 'data.npz')
        if os.path.exists(npz):
            data = np.load(npz)
            print(f"  ✓ {name:<15} shape={data['X'].shape}")
        else:
            print(f"  ✗ {name:<15} NOT READY")
    print("=" * 50)


if __name__ == '__main__':
    main()
