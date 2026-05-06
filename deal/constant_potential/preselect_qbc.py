"""
Query-by-committee pre-selection: filter MD frames by force uncertainty.
Produces All_Data.xyz for DEAL input.

Usage:
    python preselect_qbc.py \
        --std_dev_file ../MD/std_dev/MACE_Au_H2O_PZC_with_std_dev.xyz \
        --outdir . \
        --uncertainty_threshold 0.30 \
        --max_threshold_factor 3
"""

import argparse
import os
import sys

import numpy as np
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description="QbC pre-selection for DEAL")
    parser.add_argument("--std_dev_file", required=True)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.30)
    parser.add_argument("--max_threshold_factor", type=float, default=3.0)
    args = parser.parse_args()

    max_threshold = args.max_threshold_factor * args.uncertainty_threshold

    print(f"Reading {args.std_dev_file} ...")
    traj = read(args.std_dev_file, index=":", format="extxyz")
    print(f"Total configurations: {len(traj)}")

    uncertaintyx = np.array([atoms.get_array("std_dev_fx").max() for atoms in traj])
    uncertaintyy = np.array([atoms.get_array("std_dev_fy").max() for atoms in traj])
    uncertaintyz = np.array([atoms.get_array("std_dev_fz").max() for atoms in traj])
    uncertainty = np.maximum(np.maximum(uncertaintyx, uncertaintyy), uncertaintyz)

    for i, atoms in enumerate(traj):
        atoms.info["uncertainty"] = uncertainty[i]

    preselection = (uncertainty > args.uncertainty_threshold) & (uncertainty < max_threshold)
    print(
        f"Pre-selected: {preselection.sum()}/{len(traj)} frames "
        f"(threshold={args.uncertainty_threshold}, max={max_threshold} eV/A)"
    )
    print(
        f"Uncertainty stats: min={uncertainty.min():.4f}, "
        f"median={np.median(uncertainty):.4f}, max={uncertainty.max():.4f} eV/A"
    )

    if preselection.sum() == 0:
        print("ERROR: No frames passed pre-selection! Check threshold values.")
        sys.exit(1)

    idx = np.argwhere(preselection)[:, 0]
    np.random.seed(42)
    np.random.shuffle(idx)
    traj_input = [traj[i] for i in idx]

    outfile = os.path.join(args.outdir, "All_Data.xyz")
    write(outfile, traj_input)
    print(f"Wrote {len(traj_input)} frames to {outfile}")


if __name__ == "__main__":
    main()
