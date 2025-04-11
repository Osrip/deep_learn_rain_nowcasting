#!/bin/bash

# Usage: ./copy_runs.sh <sim_name1> [<sim_name2> ... <sim_nameN>]

# Check if at least one argument (sim name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <sim_name1> [<sim_name2> ... <sim_nameN>]"
  exit 1
fi

# Define the local base directory for complete runs
local_base_dir="/Users/jan/Documents/results_nowcasting/complete_runs"

# Loop through each sim name argument
for sim_name in "$@"; do
  # Create the necessary directory
  mkdir -p "$local_base_dir/$sim_name"

  # Rsync everything except checkpoint files in the model directory
  rsync -avz -e "ssh" --exclude='model/*.ckpt' bst981@134.2.168.43:"/home/butz/bst981/nowcasting_project/results/$sim_name/" "$local_base_dir/$sim_name/"

  echo "Copied complete folder of $sim_name without checkpoints"
done

echo "Done."
