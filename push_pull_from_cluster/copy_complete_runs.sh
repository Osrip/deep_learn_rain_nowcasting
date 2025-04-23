#!/bin/bash

# Usage: ./copy_runs.sh [-no_ckpt] [-gpu] <sim_name1> [<sim_name2> ... <sim_nameN>]

# Initialize flags
no_ckpt=false
gpu=false

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -no_ckpt)
      no_ckpt=true
      shift
      ;;
    -gpu)
      gpu=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Check if at least one argument (sim name) is provided after parsing options
if [ $# -lt 1 ]; then
  echo "Usage: $0 [-no_ckpt] [-gpu] <sim_name1> [<sim_name2> ... <sim_nameN>]"
  echo "  -no_ckpt    Exclude checkpoint files"
  echo "  -gpu        Copy to GPU machine directory instead"
  exit 1
fi

# Define directories based on flags
if [ "$gpu" = true ]; then
  local_base_dir="/home/jan/Programming/remote/first_CNN_on_Radolan/runs"
else
  local_base_dir="/Users/jan/Documents/results_nowcasting/complete_runs"
fi

# Loop through each sim name argument
for sim_name in "$@"; do
  # Create the necessary directory
  mkdir -p "$local_base_dir/$sim_name"

  # Prepare rsync command with or without checkpoint exclusion
  if [ "$no_ckpt" = true ]; then
    rsync_cmd="rsync -avz -e \"ssh\" --exclude='model/*.ckpt' bst981@134.2.168.43:\"/home/butz/bst981/nowcasting_project/results/$sim_name/\" \"$local_base_dir/$sim_name/\""
    echo "Copying $sim_name without checkpoints to $local_base_dir/$sim_name/"
  else
    rsync_cmd="rsync -avz -e \"ssh\" bst981@134.2.168.43:\"/home/butz/bst981/nowcasting_project/results/$sim_name/\" \"$local_base_dir/$sim_name/\""
    echo "Copying $sim_name with checkpoints to $local_base_dir/$sim_name/"
  fi

  # Execute the rsync command
  eval $rsync_cmd

  echo "Copied complete folder of $sim_name"
done

echo "Done."