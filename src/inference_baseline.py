import argparse
import json
import os
import torch
from ase.io import read
from mace.calculators import mace_mp

POLAR_MODELS = {"polar-1-s", "polar-1-m", "polar-1-l"}

from md_runner import run_nvt_md
from performance_evaluation import (
    ensure_output_dir,
    create_timestamped_run_dir,
    get_runtime_context,
    get_slurm_context,
    save_run_artifacts,
    write_run_metadata,
    write_aggregate_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MACE foundation model inference.")
    parser.add_argument(
        "--model",
        type=str,
        default="mace-matpes-r2scan-0",
        help="Foundation model name or path to a .model file. "
             "Named presets: small, medium, large, mace-matpes-r2scan-0, "
             "mace-matpes-pbe-0, etc. (default: mace-matpes-r2scan-0)",
    )
    parser.add_argument(
        "--default_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Default dtype for the model (default: float32)",
    )
    parser.add_argument(
        "--enable_cueq",
        action="store_true",
        default=True,
        help="Enable CuEq acceleration (default: True)",
    )
    parser.add_argument(
        "--no_cueq",
        action="store_true",
        default=False,
        help="Disable CuEq acceleration",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Path to input structure file (default: data/input_structures/input-structure.extxyz)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Steps between trajectory frame saves (default: 1)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override root directory for run outputs (default: data/results/<model>)",
    )
    parser.add_argument(
        "--md_length_ns",
        type=float,
        default=0.05,
        help="MD simulation length in nanoseconds (default: 0.05)",
    )
    return parser.parse_args()


def is_polar_model(model_name):
    return model_name in POLAR_MODELS


def load_foundation_model(model, device, default_dtype="float32", enable_cueq=True):
    print(f"Loading foundation model: {model} ...")
    if is_polar_model(model):
        from mace.calculators import mace_polar
        calc = mace_polar(
            model=model,
            device=device,
            default_dtype=default_dtype,
        )
    else:
        calc = mace_mp(
            model=model,
            device=device,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
        )
    print("Model ready.\n")
    return calc


def load_input_structure(structure_path, index=0):
    atoms = read(structure_path, index=index)
    print(f"\nLoaded input structure: {structure_path}")
    print(f"Atoms in structure: {len(atoms)}")
    return atoms


def model_label(model_name):
    """Derive a short label for directory naming from the model argument."""
    if os.path.isfile(model_name):
        return os.path.splitext(os.path.basename(model_name))[0]
    return model_name.replace("/", "_")


def main():
    args = parse_args()
    enable_cueq = args.enable_cueq and not args.no_cueq

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # go up from src/ to project root
    results_root = args.results_dir or os.path.join(parent_dir, "data", "results", model_label(args.model))
    run_scope = os.environ.get("SLURM_JOB_CONSTRAINT") or os.environ.get("CUDA_VISIBLE_DEVICES") or "unscoped"
    run_info = create_timestamped_run_dir(base_dir=results_root, grouping_label=run_scope)
    run_dir = run_info["run_dir"]
    out_dir = ensure_output_dir(os.path.join(run_dir, "performance_outputs"))

    runtime_context = get_runtime_context()
    slurm_context = get_slurm_context(run_script_path=os.path.join(parent_dir, "scripts", "run_mace.sh"))
    print("Runtime context:")
    print(json.dumps(runtime_context, indent=2))
    print(f"Run output directory: {run_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    structure_path = args.structure or os.path.join(parent_dir, "data", "input_structures", "input-structure.extxyz")
    input_index = 0

    model_config = {
        "model": args.model,
        "device": "cuda",
        "default_dtype": args.default_dtype,
        "enable_cueq": enable_cueq,
    }

    calc = load_foundation_model(**model_config)
    atoms = load_input_structure(structure_path=structure_path, index=input_index)

    if is_polar_model(args.model):
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.info["external_field"] = [0.0, 0.0, 0.0]

    print("Loaded structure details:")
    print(f"  - atoms: {len(atoms)}")
    print(f"  - pbc: {atoms.pbc}")
    print(f"  - cell: {atoms.cell.cellpar()}")

    md_length_ns = args.md_length_ns
    timestep_fs = 1
    temperature_K = 300
    friction = 0.01
    log_interval = args.log_interval

    nvt_input_params = {
        "md_length_ns": md_length_ns,
        "timestep_fs": timestep_fs,
        "temperature_K": temperature_K,
        "friction": friction,
        "log_interval": log_interval,
    }

    nvt_summary, nvt_records = run_nvt_md(
        calc=calc,
        atoms=atoms.copy(),
        out_dir=run_dir,
        **nvt_input_params,
    )
    nvt_artifacts = save_run_artifacts(
        run_name="nvt",
        records=nvt_records,
        summary=nvt_summary,
        out_dir=out_dir,
        include_volume=False,
    )

    aggregate_path, perf_csv_path = write_aggregate_outputs(
        out_dir=out_dir,
        runtime_context=runtime_context,
        run_summaries=[nvt_summary],
        run_artifacts={
            "nvt": nvt_artifacts,
        },
        comparison_plot=None,
    )

    metadata = {
        "run": run_info,
        "input": {
            "structure_path": structure_path,
            "input_index": input_index,
            "num_atoms": len(atoms),
            "pbc": atoms.pbc.tolist(),
            "cell_parameters": list(atoms.cell.cellpar()),
        },
        "model": model_config,
        "parameters": {
            "nvt": nvt_input_params,
        },
        "runtime_context": runtime_context,
        "slurm": slurm_context,
        "artifacts": {
            "run_dir": run_dir,
            "performance_outputs_dir": out_dir,
            "aggregate_summary": aggregate_path,
            "performance_table": perf_csv_path,
            "nvt": nvt_artifacts,
        },
    }
    metadata_path = write_run_metadata(run_dir=run_dir, metadata=metadata)

    latest_run_marker = os.path.join(results_root, "latest_run.txt")
    with open(latest_run_marker, "w", encoding="utf-8") as handle:
        handle.write(f"{run_dir}\n")

    print("Performance artifacts written to:")
    print(f"  - {run_dir}")
    print(f"  - metadata: {metadata_path}")
    print(f"  - aggregate summary: {aggregate_path}")
    print(f"  - performance table: {perf_csv_path}")


if __name__ == "__main__":
    main()
