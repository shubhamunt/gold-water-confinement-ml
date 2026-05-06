#!/usr/bin/env python
"""
Partition the dataset into 4 train/val splits for MLFF1-4.

Run from the Training/ directory:
    python partition_dataset.py

For each of the 4 models, this script:
  1. Randomly samples 70% of all frames
  2. Splits those into 85% train / 15% val
  3. Writes train_model_N.xyz and val_model_N.xyz into MLFF{N}/

Keys (energy, forces, stress, etc.) are preserved as-is from the source dataset.
"""

import os
import sys
import numpy as np
from glob import glob
from ase.io import read, write

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRACTION_4_MODEL = 0.7   # fraction of total dataset per model
FRACTION_VAL = 0.15      # fraction of model's subset used for validation
NUM_MODELS = 4
SEED = 42                # set to None for non-reproducible random splits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if SEED is not None:
        np.random.seed(SEED)

    # Find all dataset files
    xyz_files = sorted(glob(os.path.join(DATASET_DIR, "*.xyz"))) + \
                sorted(glob(os.path.join(DATASET_DIR, "*.extxyz")))

    if not xyz_files:
        print("ERROR: No .xyz or .extxyz files found in Dataset/")
        sys.exit(1)

    print(f"Found {len(xyz_files)} dataset file(s):")
    for f in xyz_files:
        print(f"  {os.path.basename(f)}")

    # Read all frames
    all_data = []
    for fpath in xyz_files:
        frames = read(fpath, index=":", format="extxyz")
        print(f"  {os.path.basename(fpath)}: {len(frames)} frames")
        all_data.extend(frames)

    n_total = len(all_data)
    print(f"\nTotal frames: {n_total}")

    if n_total == 0:
        print("ERROR: No frames found in dataset files")
        sys.exit(1)

    # Partition for each model
    for model_i in range(1, NUM_MODELS + 1):
        mlff_dir = os.path.join(SCRIPT_DIR, f"MLFF{model_i}")
        os.makedirs(mlff_dir, exist_ok=True)

        # Random sample of fraction_4_model
        n_extract = int(FRACTION_4_MODEL * n_total)
        idx_extract = np.random.choice(n_total, size=n_extract, replace=False)
        subset = [all_data[i] for i in idx_extract]

        # Split into train / val
        n_val = int(FRACTION_VAL * len(subset))
        idx_val = np.random.choice(len(subset), size=n_val, replace=False)
        idx_val_set = set(idx_val)

        train = [subset[i] for i in range(len(subset)) if i not in idx_val_set]
        val = [subset[i] for i in idx_val]

        train_file = os.path.join(mlff_dir, f"train_model_{model_i}.xyz")
        val_file = os.path.join(mlff_dir, f"val_model_{model_i}.xyz")

        write(train_file, train, format="extxyz")
        write(val_file, val, format="extxyz")

        print(f"  MLFF{model_i}: train={len(train)}, val={len(val)} "
              f"-> {os.path.relpath(train_file, SCRIPT_DIR)}, "
              f"{os.path.relpath(val_file, SCRIPT_DIR)}")

    print("\nDone! Train/val files written to MLFF1-4/")


if __name__ == "__main__":
    main()
