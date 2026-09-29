#!/bin/bash
#SBATCH --job-name=BZMass_kernel_audit
#SBATCH --partition=test
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/validation_results/bzmass_20260928/slurm_%j.log
set -eu
cd /n/home12/cgarciaquintero/DESI/src/fkptjax_muMG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SERVER_URI21 PMIX_SERVER_URI3 PMIX_SERVER_URI4 PMI_RANK PMI_SIZE PMI_FD
export FOLPS_BACKEND=jax JAX_ENABLE_X64=True JAX_PLATFORMS=cpu
python_bin=/n/home12/cgarciaquintero/.conda/envs/cosmodesi_new/bin/python
if [ "${1:-}" = reference ]; then
    /n/sw/mathematica/mathematica-15.0/Executables/wolframscript -file examples/validate_bzmass_reference.wl
    exit
fi
srun --mpi=pmix -n 1 -c "${SLURM_CPUS_PER_TASK:-4}" --cpu-bind=cores "$python_bin" -u examples/validate_bzmass_pipeline.py "${@:-kernels}"
