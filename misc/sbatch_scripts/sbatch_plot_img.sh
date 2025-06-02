#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-05:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:4    # optionally type and number of gpus
#SBATCH --partition=gpu-v100
#SBATCH --output=out_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=err_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jan-gerhard.prosi@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
# scontrol show job $SLURM_JOB_ID

# insert your commands here
#source venv/bin/activate
# python -m will run my_plotting_script.py as if it were the main module, regardless of where the current working directory is.
# The Python interpreter will handle the import statements as if the module were being imported, which means that it will
# use the package structure for resolving imports.
# Alternatively I could install my whole project with pip to resolve these conflicts. When the code is changed, it will be
# updated automatically.

python3 -m plotting.calc_and_plot_from_checkpoint

