#!/bin/bash
#SBATCH --partition=singhlab-gpu # partition
#SBATCH --account=singhlab
#SBATCH --job-name=meet    # job -name , change from command line 
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12-00:00:00 # you have asked for 12 days
#SBATCH --output=slurm_output/%x.%j.out # Standard output log, %x is the job name, %j is the job ID, y is custom time stamp
#SBATCH --error=slurm_output/%x.%j.err      # Standard error log, %x is the job name, %j is the job ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pooja.parameswaran@duke.edu

echo Job Name: meet


export TORCH_HOME='/cwork/pkp14'
python -u run_study_session.py "$@"
