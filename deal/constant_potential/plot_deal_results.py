"""
Plot DEAL selection results for Au/H2O PZC active learning.

Generates:
  1. uncertainty_histogram.png  — force uncertainty distribution with threshold
  2. deal_selection_curves.png  — cumulative selection curves per cutoff/threshold
  3. deal_summary.txt           — text summary of selection counts

Usage:
    python plot_deal_results.py \
        --std_dev_file ../MD/std_dev/MACE_Au_H2O_PZC_with_std_dev.xyz \
        --deal_dir . \
        --uncertainty_threshold 0.052 \
        --cutoffs 4.5 5.5 \
        --deal_thresholds 0.1 0.15
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from ase.io import read


def plot_uncertainty_histogram(std_dev_file, threshold, outdir):
    """Plot force uncertainty distribution with pre-selection threshold."""
    print(f"Reading {std_dev_file} ...")
    traj = read(std_dev_file, index=":", format="extxyz")

    uncertaintyx = np.array([atoms.get_array("std_dev_fx").max() for atoms in traj])
    uncertaintyy = np.array([atoms.get_array("std_dev_fy").max() for atoms in traj])
    uncertaintyz = np.array([atoms.get_array("std_dev_fz").max() for atoms in traj])
    uncertainty = np.maximum(np.maximum(uncertaintyx, uncertaintyy), uncertaintyz)

    max_threshold = 3 * threshold
    preselection = (uncertainty > threshold) & (uncertainty < max_threshold)
    print(f"Pre-selected: {preselection.sum()}/{len(traj)} frames")

    bins = np.linspace(0, max(0.15, uncertainty.max() * 1.1), 200)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(uncertainty, bins=bins, alpha=0.4, color="steelblue", label="All configurations")
    ax.hist(
        uncertainty[preselection], bins=bins, alpha=0.6, color="steelblue",
        label=f"> threshold={threshold}",
    )
    ax.axvline(threshold, color="red", ls="--", lw=1.2, label=f"threshold={threshold}")
    ax.axvline(max_threshold, color="darkred", ls=":", lw=1.2, label=f"max={max_threshold:.3f}")
    ax.set_xlabel(r"Force uncertainty $\sigma^{(\mathrm{MACE})}_{\mathrm{max}}$ [eV/$\AA$]", fontsize=12)
    ax.set_ylabel("Number of configurations", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(labelsize=11)
    fig.tight_layout()

    outpath = os.path.join(outdir, "uncertainty_histogram.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_selection_curves(deal_dir, cutoffs, deal_thresholds, outdir):
    """Plot cumulative DEAL selection curves for each cutoff/threshold combo."""
    all_data_file = os.path.join(deal_dir, "All_Data.xyz")
    if not os.path.exists(all_data_file):
        print(f"WARNING: {all_data_file} not found — skipping selection curves.")
        return

    traj_all = read(all_data_file, index=":", format="extxyz")
    n_total = len(traj_all)
    print(f"Total pre-selected frames: {n_total}")

    summary_lines = [f"Total pre-selected frames: {n_total}", ""]

    # One figure per cutoff
    for cutoff in cutoffs:
        traj_deals = {}
        for deal_thr in deal_thresholds:
            run_dir = os.path.join(deal_dir, f"threshold-{deal_thr:.3f}", f"cutoff-{cutoff}")
            selected_file = os.path.join(run_dir, "deal_selected.xyz")
            if os.path.exists(selected_file):
                traj_sel = read(selected_file, index=":", format="extxyz")
                traj_deals[deal_thr] = traj_sel
                line = f"Cutoff {cutoff}, threshold {deal_thr}: {len(traj_sel)}/{n_total} selected"
                print(line)
                summary_lines.append(line)
            else:
                line = f"Cutoff {cutoff}, threshold {deal_thr}: deal_selected.xyz NOT FOUND"
                print(line)
                summary_lines.append(line)

        if not traj_deals:
            continue

        # Use a standard colormap
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(traj_deals)))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        for k, (deal_thr, traj_sel) in enumerate(traj_deals.items()):
            try:
                selected_ids = [atoms.info["step"] for atoms in traj_sel]
            except KeyError:
                # Fallback: use index in All_Data as frame id
                selected_ids = list(range(len(traj_sel)))
            selection_curve = [
                np.sum(np.asarray(selected_ids) <= i) for i in range(n_total)
            ]
            ax.plot(
                selection_curve, label=f"DEAL-{deal_thr}",
                linewidth=2.5, alpha=0.95, color=colors[k],
            )

        ax.set_title(f"Cutoff {cutoff}", fontsize=13)
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("# Selected", fontsize=12)
        ax.legend(frameon=False, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        fig.tight_layout()

        outpath = os.path.join(outdir, f"deal_selection_curves_cutoff_{cutoff}.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"Saved {outpath}")

    # Write text summary
    summary_path = os.path.join(outdir, "deal_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot DEAL results for Au/H2O PZC")
    parser.add_argument("--std_dev_file", required=True, help="Path to std_dev extxyz file")
    parser.add_argument("--deal_dir", default=".", help="Base DEAL directory")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.052)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[4.5, 5.5])
    parser.add_argument("--deal_thresholds", type=float, nargs="+", default=[0.1, 0.15])
    parser.add_argument("--outdir", default=".", help="Where to save plots")
    parser.add_argument("--histogram_only", action="store_true",
                        help="Only plot the uncertainty histogram (before DEAL)")
    parser.add_argument("--curves_only", action="store_true",
                        help="Only plot the selection curves (after DEAL)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    run_all = not args.histogram_only and not args.curves_only

    if run_all or args.histogram_only:
        print("=== Uncertainty histogram ===")
        plot_uncertainty_histogram(args.std_dev_file, args.uncertainty_threshold, args.outdir)

    if run_all or args.curves_only:
        print("\n=== DEAL selection curves ===")
        plot_selection_curves(args.deal_dir, args.cutoffs, args.deal_thresholds, args.outdir)


if __name__ == "__main__":
    main()
