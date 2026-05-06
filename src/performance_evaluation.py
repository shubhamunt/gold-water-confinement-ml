import os
import json
import time
import csv
import re
import sys
import socket
import platform
import subprocess
from datetime import datetime, timezone
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_output_dir(path="performance_outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_label(value, default="unknown"):
    if value is None:
        return default
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return cleaned or default


def create_timestamped_run_dir(base_dir, grouping_label=None):
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    group = _sanitize_label(grouping_label or os.environ.get("SLURM_JOB_CONSTRAINT") or "unscoped")

    run_name = f"{timestamp}_job{job_id}"
    group_dir = os.path.join(base_dir, group)
    run_dir = os.path.join(group_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "group": group,
        "run_name": run_name,
        "run_dir": run_dir,
        "timestamp_utc": timestamp,
        "job_id": job_id,
    }


def _read_sbatch_directives(script_path):
    if script_path is None or not os.path.exists(script_path):
        return []

    directives = []
    with open(script_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#SBATCH"):
                directives.append(stripped)
    return directives


def _get_scontrol_details(job_id):
    if not job_id:
        return None
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", str(job_id), "-o"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {
                "available": False,
                "error": result.stderr.strip() or f"scontrol exited with code {result.returncode}",
            }
        return {
            "available": True,
            "job_o": result.stdout.strip(),
        }
    except FileNotFoundError:
        return {
            "available": False,
            "error": "scontrol not found on PATH",
        }


def get_slurm_context(run_script_path=None):
    job_id = os.environ.get("SLURM_JOB_ID")
    slurm_env = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    return {
        "job_id": job_id,
        "sbatch_directives": _read_sbatch_directives(run_script_path),
        "slurm_environment": slurm_env,
        "scontrol": _get_scontrol_details(job_id),
    }


def write_run_metadata(run_dir, metadata, file_name="run_metadata.json"):
    os.makedirs(run_dir, exist_ok=True)
    metadata_path = os.path.join(run_dir, file_name)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def summarize_series(values):
    if len(values) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    array = np.array(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def compute_energy_drift(records, timestep_fs):
    if len(records) < 2:
        return {
            "drift_total_eV": None,
            "drift_rate_eV_per_ps": None,
        }
    steps = np.array([row["step"] for row in records], dtype=float)
    etot = np.array([row["etot_eV"] for row in records], dtype=float)
    slope_eV_per_step = float(np.polyfit(steps, etot, 1)[0])
    drift_rate_eV_per_ps = slope_eV_per_step * (1000.0 / timestep_fs)
    drift_total_eV = float(etot[-1] - etot[0])
    return {
        "drift_total_eV": drift_total_eV,
        "drift_rate_eV_per_ps": float(drift_rate_eV_per_ps),
    }


def get_runtime_context():
    context = {
        "timestamp_unix_s": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if torch.cuda.is_available():
        context["cuda_device_name"] = torch.cuda.get_device_name(0)
        context["cuda_runtime_version"] = torch.version.cuda
    else:
        context["cuda_device_name"] = None
        context["cuda_runtime_version"] = torch.version.cuda
    return context


def build_nvt_summary(records, total_elapsed, temperature_k, timestep_fs, friction, n_steps, log_interval, gpu_metrics=None):
    simulated_ps = (n_steps * timestep_fs) / 1000.0
    drift = compute_energy_drift(records, timestep_fs=timestep_fs)
    summary = {
        "run_name": "nvt",
        "md_parameters": {
            "temperature_k": float(temperature_k),
            "timestep_fs": float(timestep_fs),
            "friction": float(friction),
            "n_steps": int(n_steps),
            "log_interval": int(log_interval),
        },
        "performance": {
            "wall_time_s": float(total_elapsed),
            "steps_per_second": float(n_steps / total_elapsed),
            "simulated_ps": float(simulated_ps),
            "simulated_ps_per_day": float(simulated_ps / (total_elapsed / 86400.0)),
        },
        "stability": {
            "temperature_K": summarize_series([row["temperature_K"] for row in records]),
            "pressure_GPa": summarize_series([row["pressure_GPa"] for row in records]) if records and "pressure_GPa" in records[0] else None,
            "epot_eV": summarize_series([row["epot_eV"] for row in records]),
            "ekin_eV": summarize_series([row["ekin_eV"] for row in records]),
            "etot_eV": summarize_series([row["etot_eV"] for row in records]),
            **drift,
        },
    }
    if gpu_metrics is not None:
        summary["gpu"] = gpu_metrics
    return summary


def build_npt_summary(
    records,
    total_elapsed,
    temperature_k,
    timestep_fs,
    npt_q,
    external_stress_ev_a3,
    pfactor,
    mask,
    n_steps,
    log_interval,
    gpu_metrics=None,
):
    simulated_ps = (n_steps * timestep_fs) / 1000.0
    volumes = [row["volume_A3"] for row in records]
    volume_change_pct = None
    if len(volumes) >= 2 and volumes[0] != 0:
        volume_change_pct = float(100.0 * (volumes[-1] - volumes[0]) / volumes[0])
    drift = compute_energy_drift(records, timestep_fs=timestep_fs)

    summary = {
        "run_name": "npt",
        "md_parameters": {
            "temperature_k": float(temperature_k),
            "timestep_fs": float(timestep_fs),
            "npt_q": float(npt_q),
            "external_stress_ev_a3": float(external_stress_ev_a3),
            "pfactor": float(pfactor),
            "mask": list(mask),
            "n_steps": int(n_steps),
            "log_interval": int(log_interval),
        },
        "performance": {
            "wall_time_s": float(total_elapsed),
            "steps_per_second": float(n_steps / total_elapsed),
            "simulated_ps": float(simulated_ps),
            "simulated_ps_per_day": float(simulated_ps / (total_elapsed / 86400.0)),
        },
        "stability": {
            "temperature_K": summarize_series([row["temperature_K"] for row in records]),
            "pressure_GPa": summarize_series([row["pressure_GPa"] for row in records]) if records and "pressure_GPa" in records[0] else None,
            "epot_eV": summarize_series([row["epot_eV"] for row in records]),
            "ekin_eV": summarize_series([row["ekin_eV"] for row in records]),
            "etot_eV": summarize_series([row["etot_eV"] for row in records]),
            "volume_A3": summarize_series(volumes),
            "volume_change_percent": volume_change_pct,
            **drift,
        },
    }
    if gpu_metrics is not None:
        summary["gpu"] = gpu_metrics
    return summary


def write_records_csv(file_path, records):
    if len(records) == 0:
        return
    fieldnames = list(records[0].keys())
    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_md_plot(run_name, records, out_dir, include_volume=False):
    if len(records) == 0:
        return None

    steps = [row["step"] for row in records]
    temps = [row["temperature_K"] for row in records]
    epot = [row["epot_eV"] for row in records]
    ekin = [row["ekin_eV"] for row in records]
    etot = [row["etot_eV"] for row in records]

    nrows = 3 if include_volume else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 4 * nrows), sharex=True)
    if nrows == 2:
        axes = [axes[0], axes[1]]

    axes[0].plot(steps, temps)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(f"{run_name} temperature evolution")

    axes[1].plot(steps, epot, label="Epot")
    axes[1].plot(steps, ekin, label="Ekin")
    axes[1].plot(steps, etot, label="Etot")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].legend()

    if include_volume:
        volume = [row["volume_A3"] for row in records]
        axes[2].plot(steps, volume)
        axes[2].set_ylabel("Volume (A^3)")
        axes[2].set_xlabel("MD step")
    else:
        axes[1].set_xlabel("MD step")

    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{run_name}_metrics.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def save_run_artifacts(run_name, records, summary, out_dir, include_volume=False):
    csv_path = os.path.join(out_dir, f"{run_name}_timeseries.csv")
    json_path = os.path.join(out_dir, f"{run_name}_summary.json")
    write_records_csv(csv_path, records)
    plot_path = save_md_plot(run_name, records, out_dir, include_volume=include_volume)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "timeseries_csv": csv_path,
        "summary_json": json_path,
        "metrics_plot": plot_path,
    }


def save_comparison_plot(run_summaries, out_dir):
    if len(run_summaries) == 0:
        return None

    run_names = [item["run_name"] for item in run_summaries]
    steps_per_second = [item["performance"]["steps_per_second"] for item in run_summaries]
    ps_per_day = [item["performance"]["simulated_ps_per_day"] for item in run_summaries]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].bar(run_names, steps_per_second)
    axes[0].set_ylabel("Steps / second")
    axes[0].set_title("Integrator throughput")

    axes[1].bar(run_names, ps_per_day)
    axes[1].set_ylabel("Simulated ps / day")
    axes[1].set_title("Effective simulation speed")

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "md_performance_comparison.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def print_compact_summary(summary):
    perf = summary["performance"]
    print(
        f"[{summary['run_name']}] "
        f"wall={perf['wall_time_s']:.2f}s | "
        f"steps/s={perf['steps_per_second']:.3f} | "
        f"ps/day={perf['simulated_ps_per_day']:.2f}"
    )


def write_aggregate_outputs(
    out_dir,
    runtime_context,
    run_summaries,
    run_artifacts,
    comparison_plot,
):
    artifacts = dict(run_artifacts)
    if comparison_plot is not None:
        artifacts["comparison_plot"] = comparison_plot

    aggregate = {
        "runtime_context": runtime_context,
        "runs": run_summaries,
        "artifacts": artifacts,
    }

    aggregate_path = os.path.join(out_dir, "foundation_model_performance_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    perf_csv_path = os.path.join(out_dir, "md_run_performance_table.csv")
    with open(perf_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "wall_time_s",
                "steps_per_second",
                "simulated_ps_per_day",
                "temperature_mean_K",
                "temperature_std_K",
                "pressure_mean_GPa",
                "pressure_std_GPa",
                "etot_drift_rate_eV_per_ps",
                "gpu_peak_allocated_mb",
            ],
        )
        writer.writeheader()
        for item in run_summaries:
            writer.writerow(
                {
                    "run_name": item["run_name"],
                    "wall_time_s": item["performance"]["wall_time_s"],
                    "steps_per_second": item["performance"]["steps_per_second"],
                    "simulated_ps_per_day": item["performance"]["simulated_ps_per_day"],
                    "temperature_mean_K": item["stability"]["temperature_K"]["mean"],
                    "temperature_std_K": item["stability"]["temperature_K"]["std"],
                    "pressure_mean_GPa": (item["stability"].get("pressure_GPa") or {}).get("mean"),
                    "pressure_std_GPa": (item["stability"].get("pressure_GPa") or {}).get("std"),
                    "etot_drift_rate_eV_per_ps": item["stability"]["drift_rate_eV_per_ps"],
                    "gpu_peak_allocated_mb": item.get("gpu", {}).get("peak_allocated_mb"),
                }
            )

    return aggregate_path, perf_csv_path
