# This script deploys the bash file to the cluster. Simply call it.

# This script copies a local project directory to a remote server.
# It reads the simulation name suffix from a YAML config file,
# determines the next available copy number on the remote machine
# by scanning existing "copy_*" folders, creates a new target folder
# with that number and the sim name, and syncs the local directory
# (excluding specified folders) to that remote target via rsync.

#!/bin/bash
# Path to the YAML config file
config_file="/Users/jan/Programming/first_CNN_on_Radolan/configs/cluster_default.yml"

# Extract the sim name suffix from the YAML file
sim_name_suffix=$(grep "s_sim_name_suffix:" "$config_file" | sed -e 's/.*s_sim_name_suffix: *"//' -e 's/".*//')

# Remote server details
remote_user="bst981"
remote_host="134.2.168.43"
remote_scripts_dir="/home/butz/bst981/nowcasting_project/scripts"

# Calculate the next copy number on the remote server
highest=$(ssh ${remote_user}@${remote_host} "ls -d ${remote_scripts_dir}/copy_* 2>/dev/null | sed 's|.*/copy_||; s|_.*||' | sort -n | tail -1")
if [[ -z "$highest" ]]; then
    next_number=1
else
    next_number=$((highest+1))
fi

# Build remote directory path
remote_dir="${remote_scripts_dir}/copy_${next_number}_${sim_name_suffix}"

# Create the remote directory
ssh ${remote_user}@${remote_host} "mkdir -p \"$remote_dir\""

# Local source directory
source_dir="/Users/jan/Programming/first_CNN_on_Radolan/"

# Rsync files excluding specified folders
rsync -auvh --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' \
      -e "ssh" "${source_dir}"* "${remote_user}@${remote_host}:$remote_dir"