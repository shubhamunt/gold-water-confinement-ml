"""
Set up DFT single-point calculation directories for DEAL-selected configurations.

Uses create_DFT_scripts from workflow_utils to create case_N/ folders.

Usage:
    python setup_dft_runs.py \
        --poscar_dir ../DEAL/POSCAR_temp \
        --case PZC \
        --template_script sbatch_vasp_ase_template \
        --python_calc_file DoubleReference_workflow_PZC.py
"""

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from workflow_utils.python_utils import create_DFT_scripts


def main():
    parser = argparse.ArgumentParser(description="Set up DFT single-point runs")
    parser.add_argument("--poscar_dir", required=True, help="Directory with POSCAR_0, POSCAR_1, ...")
    parser.add_argument("--case", default="PZC", help="Case label (e.g., PZC)")
    parser.add_argument("--template_script", default="sbatch_vasp_ase_template",
                        help="Template sbatch script")
    parser.add_argument("--python_calc_file", default="DoubleReference_workflow_PZC.py",
                        help="Python script for the DFT workflow")
    args = parser.parse_args()

    poscar_dir = os.path.abspath(args.poscar_dir)

    # Count POSCARs
    poscar_files = glob.glob(os.path.join(poscar_dir, "POSCAR_*"))
    num_configurations = len(poscar_files)
    if num_configurations == 0:
        print(f"ERROR: No POSCAR files found in {poscar_dir}")
        sys.exit(1)

    print(f"Found {num_configurations} POSCAR files in {poscar_dir}")

    create_DFT_scripts(
        num_configurations=num_configurations,
        case=args.case,
        template_script=args.template_script,
        path_poscar=poscar_dir,
        python_calc_file=args.python_calc_file,
        customize_potential=False,
    )

    print(f"\nCreated {num_configurations} case directories under {args.case}/")


if __name__ == "__main__":
    main()
