import time
import os
import numpy as np
import torch
from ase import units
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from performance_evaluation import (
    build_nvt_summary,
    build_npt_summary,
    print_compact_summary,
)


EV_A3_TO_GPA = 160.21766208


def _compute_pressure_gpa(atoms):
    stress = atoms.get_stress(voigt=True)
    pressure_ev_a3 = -float(np.mean(stress[:3]))
    return pressure_ev_a3 * EV_A3_TO_GPA


def get_gpu_metrics():
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
    }


def run_nvt_md(
    calc,
    atoms,
    md_length_ns=0.01,
    timestep_fs=0.5,
    temperature_K=300.0,
    friction=0.01,
    log_interval=10,
    out_dir=".",
):
    n_steps = int((md_length_ns * 1_000_000.0) / timestep_fs)
    print("\n--- Running Full MD (NVT) ---")
    print(
        f"MD settings: length={md_length_ns} ns, T={temperature_K} K, dt={timestep_fs} fs, "
        f"friction={friction} fs^-1, steps={n_steps}, log_interval={log_interval}"
    )

    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms=atoms,
        temperature_K=temperature_K,
    )

    dyn = Langevin(
        atoms=atoms,
        temperature_K=temperature_K,
        friction=friction / units.fs,
        timestep=timestep_fs * units.fs,
    )

    os.makedirs(out_dir, exist_ok=True)
    traj_file = os.path.join(out_dir, "md_nvt_foundation_model.xyz")
    thermo_log_file = os.path.join(out_dir, "md_nvt_thermo.log")

    if os.path.exists(traj_file):
        os.remove(traj_file)

    def write_xyz_frame():
        atoms.write(traj_file, format="extxyz", append=True)

    dyn.attach(write_xyz_frame, interval=1)

    md_logger = MDLogger(
        dyn=dyn,
        atoms=atoms,
        logfile=thermo_log_file,
        header=True,
        stress=True,
        peratom=False,
        mode="w",
    )
    dyn.attach(md_logger, interval=log_interval)

    start_time = time.time()
    records = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def progress_report():
        step = dyn.nsteps
        elapsed = time.time() - start_time
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        pressure_gpa = _compute_pressure_gpa(atoms)
        etot = epot + ekin
        records.append(
            {
                "step": int(step),
                "elapsed_s": float(elapsed),
                "temperature_K": float(temp),
                "pressure_GPa": float(pressure_gpa),
                "epot_eV": float(epot),
                "ekin_eV": float(ekin),
                "etot_eV": float(etot),
            }
        )
        print(
            f"[NVT] step={step:5d}/{n_steps} | "
            f"T={temp:8.2f} K | "
            f"P={pressure_gpa:8.3f} GPa | "
            f"Epot={epot:10.5f} eV | "
            f"Ekin={ekin:10.5f} eV | "
            f"Etot={etot:10.5f} eV | "
            f"elapsed={elapsed:7.2f}s"
        )

    dyn.attach(progress_report, interval=log_interval)

    print("Starting NVT integration...")
    dyn.run(n_steps)

    total_elapsed = time.time() - start_time
    summary = build_nvt_summary(
        records=records,
        total_elapsed=total_elapsed,
        temperature_k=temperature_K,
        timestep_fs=timestep_fs,
        friction=friction,
        n_steps=n_steps,
        log_interval=log_interval,
        gpu_metrics=get_gpu_metrics(),
    )

    print("NVT run complete.")
    print(f"Saved trajectory to: {traj_file}")
    print(f"Saved thermodynamic log to: {thermo_log_file}")
    print(f"Total MD wall time: {total_elapsed:.2f} s")
    print_compact_summary(summary)
    return summary, records


def run_npt_md(
    calc,
    atoms,
    temperature_k=300.0,
    timestep_fs=0.5,
    npt_q=43.06225052549201,
    external_stress_ev_a3=6000 / 1602176.6208,
    pfactor=0.1,
    mask=(0, 0, 1),
    n_steps=200,
    log_interval=10,
    out_dir=".",
):
    print("\n--- Running Full MD (NPT)---")
    print(
        f"MD settings: T={temperature_k} K, dt={timestep_fs} fs, "
        f"npt_q={npt_q}, external_stress={external_stress_ev_a3:.6f} eV/Å³, "
        f"pfactor={pfactor}, mask={mask}, steps={n_steps}, log_interval={log_interval}"
    )

    if not atoms.pbc.any():
        print("Detected non-periodic structure; adding a cubic box (15 Å) and enabling PBC for NPT.")
        atoms.center(vacuum=7.5)
        atoms.set_pbc(True)

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_k * units.kB)

    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        ttime=npt_q,
        externalstress=external_stress_ev_a3,
        pfactor=pfactor,
        mask=np.array(mask),
    )

    os.makedirs(out_dir, exist_ok=True)
    traj_file = os.path.join(out_dir, "md_npt_foundation_model.xyz")
    thermo_log_file = os.path.join(out_dir, "md_npt_thermo.log")

    if os.path.exists(traj_file):
        os.remove(traj_file)

    def write_xyz_frame():
        atoms.write(traj_file, format="extxyz", append=True)

    dyn.attach(write_xyz_frame, interval=log_interval)

    md_logger = MDLogger(
        dyn=dyn,
        atoms=atoms,
        logfile=thermo_log_file,
        header=True,
        stress=True,
        peratom=False,
        mode="w",
    )
    dyn.attach(md_logger, interval=log_interval)

    start_time = time.time()
    records = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def progress_report():
        step = dyn.nsteps
        elapsed = time.time() - start_time
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        pressure_gpa = _compute_pressure_gpa(atoms)
        vol = atoms.get_volume()
        etot = epot + ekin
        records.append(
            {
                "step": int(step),
                "elapsed_s": float(elapsed),
                "temperature_K": float(temp),
                "pressure_GPa": float(pressure_gpa),
                "epot_eV": float(epot),
                "ekin_eV": float(ekin),
                "etot_eV": float(etot),
                "volume_A3": float(vol),
            }
        )
        print(
            f"[NPT] step={step:5d}/{n_steps} | "
            f"T={temp:8.2f} K | "
            f"P={pressure_gpa:8.3f} GPa | "
            f"Epot={epot:10.5f} eV | "
            f"Ekin={ekin:10.5f} eV | "
            f"Etot={etot:10.5f} eV | "
            f"V={vol:9.3f} Å³ | "
            f"elapsed={elapsed:7.2f}s"
        )

    dyn.attach(progress_report, interval=log_interval)

    print("Starting NPT integration...")
    dyn.run(n_steps)

    total_elapsed = time.time() - start_time
    summary = build_npt_summary(
        records=records,
        total_elapsed=total_elapsed,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        npt_q=npt_q,
        external_stress_ev_a3=external_stress_ev_a3,
        pfactor=pfactor,
        mask=mask,
        n_steps=n_steps,
        log_interval=log_interval,
        gpu_metrics=get_gpu_metrics(),
    )

    print("NPT run complete.")
    print(f"Saved trajectory to: {traj_file}")
    print(f"Saved thermodynamic log to: {thermo_log_file}")
    print(f"Total MD wall time: {total_elapsed:.2f} s")
    print_compact_summary(summary)
    return summary, records
