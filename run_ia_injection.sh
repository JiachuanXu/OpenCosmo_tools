#!/bin/bash
#SBATCH --job-name=addIA
#SBATCH --account=j.blazek
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiac.xu@northeastern.edu

conda init bash
source ~/.bashrc

# set up for problem & define any environment variables here

cd /home/jiac.xu/OpenUniverse

export JAX_PLATFORMS=cpu

conda activate diffsky

export DATA_DIR="/scratch/jiac.xu/diffsky/hlwas_cosmos_260215_02_17_2026"
export OUT_DIR="/scratch/jiac.xu/diffsky/subset_catalog"

mpirun -n ${SLURM_NTASKS} python inject_ia.py --data_dir ${DATA_DIR} --output_dir ${OUTDIR} --z_min 0.0 --z_max 4.0 --central_alignment 1.0 --satellite_alignment 1.0 

# perform any cleanup or short post-processing here
