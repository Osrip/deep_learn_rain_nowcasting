#!/bin/bash

#Usage: ./copy_runs.sh <sim_name1> [<sim_name2> ... <sim_nameN>]

# Check if at least one argument (sim name) is provided


#if [ $# -lt 1 ]; then
#  echo "Usage: $0 <sim_name1> [<sim_name2> ... <sim_nameN>]"
#  exit 1
#fi
#
## Loop through each sim name argument
#for sim_name in "$@"; do
#  mkdir -p "/home/jan/Documents/results_nowcasting/$sim_name" &&
#  rsync -avz -e "ssh" bst981@134.2.168.52:"/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$sim_name" "/home/jan/Documents/results_nowcasting/complete_runs/"
#  echo "Copied complete folder of $sim_name"
#done
#
#echo "Done."


#!/bin/bash

#Usage: ./copy_runs.sh <sim_name1> [<sim_name2> ... <sim_nameN>]

# Function to find the checkpoint with the lowest loss
get_lowest_loss_checkpoint() {
  ssh bst981@134.2.168.52 "
    ls /mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$1/model/*.ckpt |
    sort -t '=' -k3,3 -n |
    head -n 1
  "
}

# Check if at least one argument (sim name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <sim_name1> [<sim_name2> ... <sim_nameN>]"
  exit 1
fi

# Define the local base directory for complete runs
local_base_dir="/home/jan/Documents/results_nowcasting/complete_runs"

# Loop through each sim name argument
for sim_name in "$@"; do
  # Create the necessary directories
  mkdir -p "$local_base_dir/$sim_name/model" &&

  # Rsync everything except the model dir

  rsync -avz -e "ssh" --exclude='/model/' bst981@134.2.168.52:"/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$sim_name/" "$local_base_dir/$sim_name/" &&


  # Rsync everything in the model directory
  rsync -avz -e "ssh" --include='*/' --include='model_epoch=0010_*.ckpt' --include='model_epoch=0049_*.ckpt' --exclude='*.ckpt' bst981@134.2.168.52:"/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$sim_name/model/" "$local_base_dir/$sim_name/model" &&

  # Get the checkpoint with the lowest loss
  lowest_loss_checkpoint=$(get_lowest_loss_checkpoint "$sim_name") &&

  # If a checkpoint with the lowest loss exists, rsync it
  if [[ -n "$lowest_loss_checkpoint" ]]; then
    rsync -avz -e "ssh" bst981@134.2.168.52:"$lowest_loss_checkpoint" "$local_base_dir/$sim_name/model/"
  fi

  echo "Copied complete folder of $sim_name with selected checkpoints"
done

echo "Done."



