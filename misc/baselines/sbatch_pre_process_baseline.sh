#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=15-00:00            # Runtime in D-HH:MM
#SBATCH --partition=cpu-galvani
#SBATCH --output=out_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=err_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jan-gerhard.prosi@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
# scontrol show job $SLURM_JOB_ID

# insert your commands here
#source venv/bin/activate
python3 pre_process_baseline.py

