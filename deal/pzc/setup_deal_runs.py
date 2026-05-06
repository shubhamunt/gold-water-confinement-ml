"""
Generate input.yaml and sbatch script for each DEAL cutoff/threshold combo.

Usage:
    python setup_deal_runs.py \
        --data_file All_Data.xyz \
        --outdir . \
        --cutoffs 4.5 5.5 \
        --deal_thresholds 0.1 0.15 \
        --mem 64G

Then submit each job individually:
    cd threshold-0.100/cutoff-4.5 && sbatch sbatch_deal && cd ../..
"""

import argparse
import os


YAML_TEMPLATE = """\
data:
  files: ["{data_file}"]
  format: extxyz
  index: ":"
  shuffle: false
  seed: 42

deal:
  threshold: {threshold}
  max_atoms_added: 0.2
  output_prefix: deal
  force_only: true
  train_hyps: false
  verbose: true
  save_gp: false

flare:
  gp: SGP_Wrapper
  kernels:
    - name: NormalizedDotProduct
      sigma: 2
      power: 2
  descriptors:
    - name: B2
      nmax: 8
      lmax: 3
      cutoff_function: cosine
      radial_basis: chebyshev
  cutoff: {cutoff}
"""

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem={mem}
#SBATCH --job-name='DEAL_cut_{cutoff}_thr_{threshold}'
#SBATCH --account=pi_ozguryilmaze_umass_edu
#SBATCH --output=deal_%j.out

module purge
module load conda/latest
conda activate {conda_env}

deal -c input.yaml

echo $(date)
echo "Job finished"
"""


def main():
    parser = argparse.ArgumentParser(description="Setup DEAL runs for each cutoff/threshold combo")
    parser.add_argument("--data_file", required=True, help="Path to All_Data.xyz")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[4.5, 5.5])
    parser.add_argument("--deal_thresholds", type=float, nargs="+", default=[0.1, 0.15])
    parser.add_argument("--mem", default="64G", help="Memory per job (default: 64G)")
    parser.add_argument(
        "--conda_env",
        default="/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/flare_deal_env",
    )
    args = parser.parse_args()

    data_file = os.path.abspath(args.data_file)

    for cutoff in args.cutoffs:
        for threshold in args.deal_thresholds:
            run_dir = os.path.join(args.outdir, f"threshold-{threshold:.3f}", f"cutoff-{cutoff}")
            os.makedirs(run_dir, exist_ok=True)

            # Symlink All_Data.xyz into run_dir (like notebook's copy_traj=False)
            link_path = os.path.join(run_dir, "All_Data.xyz")
            if not os.path.exists(link_path):
                os.symlink(data_file, link_path)

            # Write input.yaml
            yaml_path = os.path.join(run_dir, "input.yaml")
            with open(yaml_path, "w") as f:
                f.write(YAML_TEMPLATE.format(
                    data_file=data_file,
                    cutoff=cutoff,
                    threshold=threshold,
                ))

            # Write sbatch script
            sbatch_path = os.path.join(run_dir, "sbatch_deal")
            with open(sbatch_path, "w") as f:
                f.write(SBATCH_TEMPLATE.format(
                    mem=args.mem,
                    cutoff=cutoff,
                    threshold=threshold,
                    conda_env=args.conda_env,
                ))

            print(f"Created: {run_dir}/input.yaml + sbatch_deal")

    print(f"\nTo submit all jobs:")
    for cutoff in args.cutoffs:
        for threshold in args.deal_thresholds:
            run_dir = os.path.join(args.outdir, f"threshold-{threshold:.3f}", f"cutoff-{cutoff}")
            print(f"  cd {run_dir} && sbatch sbatch_deal && cd -")


if __name__ == "__main__":
    main()
