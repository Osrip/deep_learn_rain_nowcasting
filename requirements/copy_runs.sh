#!/bin/bash

# Check if at least one argument (sim name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <sim_name1> [<sim_name2> ... <sim_nameN>]"
  exit 1
fi

# Loop through each sim name argument
for sim_name in "$@"; do
  mkdir -p "/home/jan/Documents/results_nowcasting/$sim_name/plots" &&
  rsync -avz -e "ssh" bst981@134.2.168.52:"/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$sim_name/plots" "/home/jan/Documents/results_nowcasting/$sim_name"
  echo "Copied $sim_name"

done

echo "Done."
