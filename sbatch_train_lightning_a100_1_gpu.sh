#!/bin/bash

#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --partition=a100-galvani
#SBATCH --output=out_%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=err_%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jan-gerhard.prosi@uni-tuebingen.de   # Email to which notifications will be sent

source ~/.bashrc

# Now activate the environment
conda activate /home/butz/bst981/.conda/first_CNN_on_Radolan_10/first_CNN_on_Radolan_10

# Run the Python script
srun python3 train_lightning.py --mode cluster