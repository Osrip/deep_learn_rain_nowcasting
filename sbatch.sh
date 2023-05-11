#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-24:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1    # optionally type and number of gpus
#SBATCH --partition=gpu-v100
#SBATCH --mem=200G                # Memory pool for all cores (see also --mem-per-cpu) max 400 v-100
#SBATCH --output=out_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=err_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jan-gerhard.prosi@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
# scontrol show job $SLURM_JOB_ID

# insert your commands here
#source venv/bin/activate
python3 train.py

