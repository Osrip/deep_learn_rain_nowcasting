#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:20            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1    # optionally type and number of gpus
#SBATCH --partition=gpu-2080ti
#SBATCH --mem=36G                # Memory pool for all cores (see also --mem-per-cpu) max 400 v-100
#SBATCH --output=out_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=err_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jan-gerhard.prosi@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
# scontrol show job $SLURM_JOB_ID

# insert your commands here
#source venv/bin/activate
python3 train_lightning.py

