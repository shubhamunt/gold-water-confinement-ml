from ase.io import read,write
from ase.calculators.vasp import Vasp
from ase.calculators.DoubleReferenceMethod.DoubleReferenceWorkflow_calc import DoubleReferenceEvaluator,DoubleReferenceWorkflow
import numpy as np

# Constant-potential parameters for Au/H2O at m05 (-0.5V vs SHE)
# Adjust EXTRA_ELEC after the first DFT run converges (check Bader charges)
V_VECTOR   = [-0.5]   # external bias potentials to evaluate (V vs SHE)
EXTRA_ELEC = 0        # initial guess of extra electrons; update after first convergence
C_START    = 1/80     # initial capacitance guess e/(V·Å²) — DRM default

#################   Definition of the different ASE-VASP Calculators to compute the "Double Reference Method"   #################

#### 1) No extra charge + no vacuum
calc_neutral_no_vacuum=Vasp(directory='neutral',
            istart = 0,   #restart from scratch
            icharg = 2,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1,
            algo = 'fast',
            ediff = 1E-07,
            nelm = 160,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 4,   # 4 cores/orbital → matches 32 MPI ranks on Unity (kpar=2 × ncore=4 × 4 = 32)
            kpar=2,       # 2 k-point groups (matches 2 irreducible k-points for (2,2,1) mesh)
            nsim=8,       # ~5-15% speedup for RMM-DIIS
            lwave=False,  # skip WAVECAR write for step 1 (no restart needed)
            lreal='Auto',
            laechg = True,
            lvhar   = True)

#### 2) No extra charge + vacuum

####### 2.1) No extra charge + vacuum + no dipole corrections
calc_neutral_vacuum_no_dipole=Vasp(directory='neutral_vacuum',
            istart = 0,   #restart from scratch
            icharg = 2,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1,
            algo = 'N',
            ediff = 1E-07,
            nelm = 400,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 4,
            kpar=2,
            lreal='Auto')

####### 2.2) No extra charge + vacuum + dipole corrections
calc_neutral_vacuum_dipole=Vasp(directory='neutral_vacuum',
            istart = 1,   #restart
            icharg = 1,
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1,
            algo = 'A',
            ediff = 1E-07,
            nelm = 300,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore  = 4,
            kpar=2,
            lreal='Auto',
            ldipol  = True, #switch on dipole correction
            idipol  = 3,    #in 3rd direction
            dipol   = [0.5, 0.5, 0.5],
            lvhar   = True, # to print only ionic and hartree potential
            lvacpotav=True )

#### 3) Extra charge + no vacuum
calc_charge=Vasp(directory='charge',
            prec= 'Accurate',
            encut  =  800,
            pp='PBE',
            kpts=(2, 2, 1),
            ismear = 0,
            sigma = 0.1,
            algo = 'fast',
            ediff = 1E-07,
            nelm = 160,
            lmixtau= True,
            metagga = 'R2SCAN',
            lasph = True,
            ibrion = -1,
            ncore = 4,
            kpar=2,
            lreal='Auto',
            laechg = True,
            lvhar   = True # to print only ionic and hartree potential
            )

#################   Start calculation of the "Double Reference Method" workflow   #################

snap=read('POSCAR',format='vasp')

# Prepare arguments for your workflow
workflow_args = dict(
                        external_bias_vector=V_VECTOR,
                        calc_neutral_no_vacuum=calc_neutral_no_vacuum,
                        calc_neutral_vacuum_no_dipole=calc_neutral_vacuum_no_dipole,
                        calc_neutral_vacuum_dipole=calc_neutral_vacuum_dipole,
                        calc_charge=calc_charge,
                        guess_extra_electrons=EXTRA_ELEC,
                        C_guess=C_START,
                        restart=False
                        )

# Attach evaluator
snap.evaluator = DoubleReferenceEvaluator(DoubleReferenceWorkflow, **workflow_args)

# Trigger calculation
result = snap.evaluator.evaluate(snap)

print("Calculation terminated")
