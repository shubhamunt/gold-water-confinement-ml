# gold-water-confinement-ml

Probing space confinement effects in gold nanoparticles: AIMD training and MACE-based MD simulations of confined water.

This repository contains the active learning workflow for developing machine learning force fields (MLFFs) for Au(111)/H₂O interfaces, targeting the point of zero charge (PZC) and constant-potential electrochemical conditions.

---

## Workflow overview

The workflow follows an iterative active learning loop:

```
Training → MD → DEAL (uncertainty) → DFT (labelling) → repeat
```

1. **Training** — Train an ensemble of 4 MACE models on the current dataset
2. **MD** — Run exploratory NVT molecular dynamics with LAMMPS
3. **DEAL** — Select new configurations via uncertainty-based active learning (Query-by-Committee + DEAL/FLARE)
4. **DFT** — Label selected configurations with VASP (R2SCAN for PZC, RPBE for constant potential)
5. **Repeat** — Add DFT results to the dataset and retrain

Two system variants are included:

| Variant | Functional | Description |
|---|---|---|
| `pzc/` | R2SCAN | Au(111)/H₂O at point of zero charge |
| `constant_potential/` | RPBE | Au(111)/H₂O at constant electrode potential (Franken GNN) |

---

## Repository structure

```
gold-water-confinement-ml/
├── training/           # MACE and Franken GNN training scripts
│   ├── pzc/            # 4-model MACE ensemble (R2SCAN)
│   └── constant_potential/  # Franken GNN (RPBE)
├── md/                 # LAMMPS MD input files and sbatch scripts
│   ├── pzc/
│   └── constant_potential/
├── deal/               # DEAL active learning selection
│   ├── pzc/
│   └── constant_potential/
├── dft/                # VASP DFT single-point calculations
│   ├── pzc/            # Double Reference Method (R2SCAN)
│   └── constant_potential/  # Double Reference Method (RPBE)
├── utils/              # Shared Python utilities (DFT job setup, std_dev analysis)
├── src/                # Inference baselines, MD runner, performance evaluation
├── scripts/            # Top-level bash execution scripts
├── notebooks/          # Detailed workflow notebook (WORKFLOW_PZC.ipynb)
├── environments/       # Conda environment YAML exports
└── data/
    ├── pzc/
    │   ├── dataset_Au_H2O_111_PZC_round1.xyz   # Training dataset (Round 1)
    │   └── models/MLFF{1-4}/                   # Trained MACE models
    └── constant_potential/                      # Franken training data
```

---

## Quick start

### 1. Setup

See [SETUP.md](SETUP.md) for full instructions on:
- Loading HPC modules
- Creating conda environments
- Installing external tools (Bader, VTS scripts, MacroDensity)

### 2. PZC workflow (R2SCAN, MACE ensemble)

All commands run from the repo root. See `notebooks/WORKFLOW_PZC.ipynb` for detailed explanations.

**Step 1 — Train 4 MACE models:**
```bash
cd training/pzc/
bash partition_dataset.sh
cd MLFF1 && sbatch single_gpu_train.sh && cd ..
cd MLFF2 && sbatch single_gpu_train.sh && cd ..
cd MLFF3 && sbatch single_gpu_train.sh && cd ..
cd MLFF4 && sbatch single_gpu_train.sh && cd ..
cd ../..
```
*Conda env:* `baseline_mace_env` | *GPU:* L40S 48GB | *Time:* ~20 hours

**Step 2 — Run MD:**
```bash
# Convert model to LAMMPS format
cp data/pzc/models/MLFF1/MACE_Au_H2O_PZC_1_run_stagetwo_compiled.model md/pzc/
cd md/pzc/
bash Mace2Lammps.sh /path/to/baseline_mace_env MACE_Au_H2O_PZC_1_run_stagetwo_compiled.model
sbatch sbatch_lammps_mace
cd ../..
```
*Module:* `LAMMPS/28Oct2024-foss-2023a-kokkos-mace-CUDA-12.1.1` | *Time:* ~1000 ps

**Step 3 — DEAL uncertainty selection:**
```bash
# Evaluate 4-model ensemble uncertainty on MD trajectory
cd md/pzc/std_dev/ && sbatch sbatch_evaluation_std_dev && cd ../../..

# Run DEAL selection
cd deal/pzc/
sbatch sbatch_preselect                    # QbC pre-selection
bash sbatch_setup_and_submit_deal          # Generate DEAL inputs + submit
sbatch sbatch_summarize                    # Plot + export POSCARs
cd ../..
```
*Conda env:* `flare_deal_env` | *Memory:* 256 GB (DEAL scales O(n²))

**Step 4 — DFT labelling:**
```bash
cd dft/pzc/
bash sbatch_setup_and_submit_dft           # Set up case dirs + submit all VASP jobs
sbatch sbatch_check_and_collect           # Check convergence + aggregate results
cd ../..
```
*Conda env:* `dft_vasp_env` | *Module:* `vasp/6.5.1` | *Time:* ~3h per case (Zen4)

**Repeat** from Step 1 with the updated `data/pzc/dataset_Au_H2O_111_PZC_roundN.xyz`.

### 3. Constant potential workflow (RPBE, Franken GNN)

The constant potential variant follows the same stage order. Entry points:

```bash
bash scripts/run_constant_potential.sh
```

---

## Data and models

Pre-trained models and the Round 1 dataset are in `data/`:

| File | Description |
|---|---|
| `data/pzc/dataset_Au_H2O_111_PZC_round1.xyz` | 196 R2SCAN-labelled Au/H₂O configurations |
| `data/pzc/models/MLFF{1-4}/*.model` | 4 MACE ensemble models (Round 1) |
| `data/constant_potential/Train_m05_union_with_head.xyz` | Franken training set |
| `data/constant_potential/Valid_m05_union_with_head.xyz` | Franken validation set |

**System:** Au(111) 6-layer slab + H₂O (72 Au + 56 O + 112 H = 240 atoms total)

**Training metrics (Round 1 MACE models):**
- Force RMSE: ~14.6 meV/Å
- Energy RMSE: ~0.2 meV/atom

---

## Dependencies

| Dependency | Purpose | Installation |
|---|---|---|
| [MACE](https://github.com/ACEsuit/mace) | ML force field training | `baseline_mace_env` |
| [LAMMPS](https://www.lammps.org/) | MD simulations | HPC module |
| [FLARE](https://github.com/mir-group/flare) v1.3.3b | GP uncertainty (DEAL backend) | `flare_deal_env` |
| [DEAL](https://github.com/luigibonati/DEAL) | Active learning selection | `flare_deal_env` |
| [VASP](https://www.vasp.at/) 6.5.1 | DFT single-point calculations | HPC module (licensed) |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atomic simulation interface | `dft_vasp_env` |
| [DoubleReferenceMethod](https://github.com/michelegiovannibianchi/DoubleReferenceMethod-FCP-calculator) | Electrochemical DFT workflow | `dft_vasp_env` |
| [Bader](http://theory.cm.utexas.edu/henkelman/code/bader/) | Charge density analysis | Manual install |
| [MacroDensity](https://github.com/WMD-group/MacroDensity) | Planar-averaged potential | `dft_vasp_env` |

See [SETUP.md](SETUP.md) for full installation instructions.

---

## Key notes

- **DEAL memory:** Requires `--mem=256G`. FLARE's SGP scales O(n²) in inducing points; 64 GB is insufficient after ~150 selected frames.
- **VASP CPU-only:** `vasp/6.5.1` on Unity is CPU-only. Use Zen4 nodes (32 MPI ranks, ~3h per 240-atom R2SCAN calculation). Zen4 is ~41% faster than Ice Lake for this workload.
- **DEAL YAML key:** The FLARE config key in `input.yaml` must be `flare:` (not `flare_calc:`). DEAL's CLI reads `cfg_dict["flare"]`; the wrong key causes all DEAL runs to silently use default parameters.
- **Round 1 selection rate:** ~62% selection rate in DEAL iteration 1 is normal — the MACE ensemble has high uncertainty on MD frames from a domain it has not seen during training. Rates drop significantly in later rounds.

---

## Reference

This workflow is based on the [TRECI](https://github.com/michelegiovannibianchi/TRECI) methodology for transfer learning on electrochemical interfaces (Bianchi, Bonati, Cicero et al.).
