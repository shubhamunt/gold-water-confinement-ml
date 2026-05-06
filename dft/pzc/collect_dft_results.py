"""
Collect DFT results into a single ext-xyz dataset file and copy to Training/Dataset/.

Usage:
    python collect_dft_results.py --case PZC --round_number 1
"""

import argparse
import glob
import os
import shutil
import sys
from natsort import natsorted

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description="Collect DFT results into dataset")
    parser.add_argument("--case", default="PZC", help="Case label")
    parser.add_argument("--round_number", type=int, required=True, help="Active learning round number")
    parser.add_argument("--outcar_name", default="OUTCAR_neutral_no_vaccum.xyz",
                        help="Name of the OUTCAR xyz file")
    args = parser.parse_args()

    name_dataset = f"dataset_Au_H2O_111_{args.case}_round{args.round_number}.xyz"

    file_paths = natsorted(glob.glob(f"{args.case}/case_*/case_*/"))

    if not file_paths:
        print(f"No calculation directories found under {args.case}/case_*/case_*/")
        sys.exit(1)

    print(f"Collecting results from {len(file_paths)} calculations...")

    count = 0
    for file_path in file_paths:
        outcar_xyz = os.path.join(file_path, args.outcar_name)
        if not os.path.exists(outcar_xyz):
            print(f"  WARNING: {outcar_xyz} not found, skipping")
            continue
        data = read(outcar_xyz, format="extxyz")
        write(name_dataset, data, format="extxyz", append=True)
        count += 1

    print(f"Collected {count} configurations into {name_dataset}")

    # Copy to Training/Dataset/
    dataset_dir = os.path.join("..", "Training", "Dataset")
    if os.path.isdir(dataset_dir):
        dest = os.path.join(dataset_dir, name_dataset)
        shutil.copy(name_dataset, dest)
        print(f"Copied to {dest}")
    else:
        print(f"WARNING: {dataset_dir} not found — dataset not copied to Training/Dataset/")

    print("\nReady to start a new round of active learning!")


if __name__ == "__main__":
    main()
