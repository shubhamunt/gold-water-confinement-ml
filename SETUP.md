# Setup Guide

This guide covers all dependencies and environment setup required to run the Au/H₂O active learning workflow on the **Unity HPC cluster** (UMass Amherst).

---

## 1. System Modules (Unity HPC)

Each workflow stage requires specific modules. Load them at the start of the relevant SLURM jobs or interactive sessions:

```bash
module load conda/latest           # all stages — required for conda environments

# DFT (VASP)
module load mpich/4.2.1
module load vasp/6.5.1

# MACE training
module load cuda/12.1

# LAMMPS MD
module load uri/main
module load LAMMPS/28Oct2024-foss-2023a-kokkos-mace-CUDA-12.1.1
```

> **Note:** The sbatch scripts in this repo already include the correct `module load` lines for each stage. You generally do not need to load modules manually unless running interactively.

---

## 2. Conda Environments

The workflow uses four separate conda environments. There is one per stage:

| Environment | Stage | Key packages |
|---|---|---|
| `baseline_mace_env` | Training (MACE) | mace-torch, torch (CUDA 12.1), ase |
| `flare_deal_env` | DEAL selection | flare==1.3.3b, DEAL, ase, chemiscope |
| `dft_vasp_env` | DFT (VASP + ASE) | ase==3.28.0, DoubleReferenceMethod, macrodensity |
| `franken_env` | Franken GNN training | torch (CUDA 12.1), franken, ase |

All conda environments are installed at:
```
/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/
```

### Option A — Restore from exported YAMLs (recommended)

```bash
CONDA_ENVS=/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs

conda env create -f environments/dft_vasp_env.yml      -p $CONDA_ENVS/dft_vasp_env
conda env create -f environments/flare_deal_env.yml    -p $CONDA_ENVS/flare_deal_env
conda env create -f environments/baseline_mace_env.yml -p $CONDA_ENVS/baseline_mace_env
conda env create -f environments/franken_env.yml       -p $CONDA_ENVS/franken_env
```

### Option B — Manual build (if YAML restore fails)

**`dft_vasp_env`** — Python 3.11, ASE + VASP + DoubleReferenceMethod:
```bash
CONDA_ENVS=/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs
conda create -p $CONDA_ENVS/dft_vasp_env python=3.11
conda activate $CONDA_ENVS/dft_vasp_env

pip install ase==3.28.0 pandas natsort macrodensity

git clone https://github.com/michelegiovannibianchi/DoubleReferenceMethod-FCP-calculator.git
pip install -e DoubleReferenceMethod-FCP-calculator
```

**`flare_deal_env`** — Python 3.12, FLARE + DEAL:
```bash
conda create -p $CONDA_ENVS/flare_deal_env python=3.12
conda activate $CONDA_ENVS/flare_deal_env

conda install -y gcc gxx cmake openmp liblapacke openblas -c conda-forge

git clone https://github.com/mir-group/flare.git -b 1.3.3b
cd flare && pip install . && cd ..

git clone https://github.com/luigibonati/DEAL.git
cd DEAL && pip install . && cd ..

pip install ase pandas numpy chemiscope
```

**`baseline_mace_env`** — Python 3.11, MACE + CUDA 12.1:
```bash
conda create -p $CONDA_ENVS/baseline_mace_env python=3.11
conda activate $CONDA_ENVS/baseline_mace_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mace-torch ase
```

**`franken_env`** — Python 3.11, Franken GNN:
```bash
conda create -p $CONDA_ENVS/franken_env python=3.11
conda activate $CONDA_ENVS/franken_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install franken from source (see Franken documentation)
pip install ase
```

---

## 3. External Tools

These tools are **not included in this repo** and must be installed manually on the HPC system. The expected install paths are hard-coded in the sbatch scripts.

### Bader Charge Analysis

Used in: DFT stage (charge density analysis after each VASP calculation)

```bash
# Download the binary from: http://theory.cm.utexas.edu/henkelman/code/bader/
# Install at:
mkdir -p /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/bader
# Place the binary there and make it executable:
chmod +x .../tools/bader/bader
```

The sbatch scripts reference this binary as:
```
/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/bader/bader
```

### VTS Scripts (chgsum.pl — required for Bader)

Used in: DFT stage (summing CHGCAR files before Bader analysis)

```bash
git clone https://github.com/vtstcode/vtstscripts.git \
    /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/vtstscripts-1040
```

The sbatch scripts reference `chgsum.pl` at:
```
/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/vtstscripts-1040/chgsum.pl
```

### MacroDensity (planar density averaging)

Used in: DFT stage (planar-averaged electrostatic potential for Double Reference Method)

```bash
git clone https://github.com/WMD-group/MacroDensity.git \
    /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/MacroDensity

conda activate $CONDA_ENVS/dft_vasp_env
pip install -e .../tools/MacroDensity
# Or simply: pip install macrodensity
```

---

## 4. VASP Pseudopotentials

VASP POTCAR files are institution-licensed and not redistributable. They are available on Unity at:

```
/work/pi_azagalskaya_umass_edu/tools/vasp6/potpaw_PBE/
```

The DFT sbatch scripts and ASE workflow set:
```bash
export VASP_PP_PATH=/work/pi_azagalskaya_umass_edu/tools/vasp6
```

If you are setting up on a different machine, point `VASP_PP_PATH` to your local POTCAR directory.

---

## 5. Hardcoded Paths to Update

The sbatch scripts and Python files contain absolute paths specific to the original workspace. If you clone to a different location, update these references:

| What | Current value |
|---|---|
| Conda env prefix | `/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/<env>/` |
| VASP pseudopotentials | `/work/pi_azagalskaya_umass_edu/tools/vasp6` |
| Bader binary | `.../tools/bader/bader` |
| VTS chgsum.pl | `.../tools/vtstscripts-1040/chgsum.pl` |

A quick way to find all occurrences:
```bash
grep -r "anshitagupta_umass_edu" scripts/ dft/ deal/ training/ md/ --include="sbatch_*" -l
grep -r "aravadikeshr_umass_edu" scripts/ dft/ deal/ training/ md/ --include="sbatch_*" -l
```

---

## 6. Verify Your Setup

After installing all dependencies, run a quick sanity check:

```bash
# Check MACE
conda activate $CONDA_ENVS/baseline_mace_env
python -c "import mace; print('MACE OK')"

# Check ASE + DoubleReferenceMethod
conda activate $CONDA_ENVS/dft_vasp_env
python -c "import ase; print('ASE', ase.__version__); from DoubleReferenceMethod import FCP; print('DRM OK')"

# Check DEAL + FLARE
conda activate $CONDA_ENVS/flare_deal_env
python -c "import flare; import deal; print('FLARE + DEAL OK')"

# Check Bader binary
/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/tools/bader/bader --version
```
