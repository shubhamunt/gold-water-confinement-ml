"""
Check convergence of DFT single-point calculations.

Scans OUTCAR files in DFT/PZC/case_*/case_*/ for VASP convergence status.

Usage:
    python check_dft_convergence.py --case PZC
"""

import argparse
import glob
import os
import sys
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description="Check DFT convergence")
    parser.add_argument("--case", default="PZC", help="Case label")
    parser.add_argument("--outcar_name", default="OUTCAR_neutral_no_vaccum",
                        help="Name of OUTCAR file to check")
    args = parser.parse_args()

    file_paths = natsorted(glob.glob(f"{args.case}/case_*/case_*/"))

    if not file_paths:
        print(f"No calculation directories found under {args.case}/case_*/case_*/")
        sys.exit(1)

    print(f"Check convergence ({len(file_paths)} calculations):\n")

    failed = []
    for path in file_paths:
        outcar = os.path.join(path, args.outcar_name)
        if not os.path.exists(outcar):
            print(f"  OUTCAR not found in {path}")
            failed.append(path)
            continue

        with open(outcar, "r") as f:
            end = False
            unconverged = False
            for line in f:
                if "aborting" in line:
                    end = True
                    if "unconverged" in line:
                        unconverged = True

        if unconverged:
            print(f"  UNCONVERGED: {path}")
            failed.append(path)
        elif not end:
            print(f"  DID NOT FINISH: {path}")
            failed.append(path)

    print(f"\n{len(file_paths) - len(failed)}/{len(file_paths)} converged successfully.")
    if failed:
        print(f"\n{len(failed)} problematic calculations:")
        for p in failed:
            print(f"  {p}")
        print("\nCheck and eventually remove the folders where VASP did not converge or did not finish smoothly.")


if __name__ == "__main__":
    main()
