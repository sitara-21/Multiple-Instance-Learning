#!/bin/bash
#SBATCH -J ABMIL
#SBATCH -n 8
#SBATCH -t 3-00:00:00
#SBATCH -p medium
#SBATCH -o /n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/slurm_output/abmil.%J.out
#SBATCH -e /n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/slurm_outputt/abmil.%J.err
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=asingh46@mgh.harvard.edu

source /n/app/miniconda3/23.1.0/etc/profile.d/conda.sh 
conda activate /n/data1/hms/dbmi/gulhan/lab/ankit/conda_envs/SNVCurate
module load gcc java/jdk-1.8u112

python /n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/main.py