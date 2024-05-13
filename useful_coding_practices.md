
#### Slurm
scontrol show job $SLURM_JOB_ID
scancel -jobid- 

SSD direktory to work on!
/mnt/qb/work2/butz1/bst981

#### Torch module dicts
- They can be used 
Example in https://github.com/jthuemmel/SpatioTemporalNetworks/blob/main/models/models.py
self.decoder_layers = nn.ModuleDict()
self.decoder_layers[f'scale_{i}'](z)

#### Specific stuff for this implementation
Activate remote python venv:
source /home/jan/Programming/remote/first_CNN_on_radolan_remote/virtual_env/bin/activate

For some reason has to be started without sudo!

Old:
% rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' -e ssh $(pwd)/* bst981@134.2.168.52:/mnt/qb/butz/bst981/first_CNN_on_Radolan
ifoe

##### Upload code to remote
rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' -e ssh $(pwd)/* bst981@134.2.168.52:/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan

Upload directly from correct dir
rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' -e ssh /home/jan/jan/programming/first_CNN_on_Radolan/* bst981@134.2.168.52:/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan

Upload to copy_folder
copy_num=1; rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' -e ssh /home/jan/jan/programming/first_CNN_on_Radolan/* bst981@134.2.168.52:/mnt/qb/work2/butz1/bst981/radolan_copies/copy_$copy_num

BEST UPLOAD TO REMOTE --> Upload to copy folder and create dir in case it does not exist:
LINUX
copy_num=1; remote_dir="/mnt/qb/work2/butz1/bst981/radolan_copies/copy_$copy_num"; ssh_command="ssh"; source_dir="/home/jan/jan/programming/first_CNN_on_Radolan/"; $ssh_command bst981@134.2.168.52 "mkdir -p $remote_dir"; rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' -e $ssh_command "$source_dir"* "bst981@134.2.168.52:$remote_dir"
MAC
copy_num=25; remote_dir="/mnt/qb/work2/butz1/bst981/radolan_copies/copy_$copy_num"; ssh_command="ssh"; source_dir="/Users/jan/Programming/first_CNN_on_Radolan/"; $ssh_command bst981@134.2.168.52 "mkdir -p $remote_dir"; rsync -auvh --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' --exclude 'mlruns' --exclude 'lightning_logs' -e $ssh_command "$source_dir"* "bst981@134.2.168.52:$remote_dir"

Execute sbatch on server on according number:
DOES NOT WORK FOR SOME REASON, start in folder!
copy_num=1; sbatch /mnt/qb/work2/butz1/bst981/radolan_copies/copy_$copy_num/sbatch_train_lightning.sh

##### Download plots from remote to local
sim_name=""; mkdir -p "/home/jan/Documents/results_nowcasting/$sim_name/plots" && rsync -avz -e "ssh" bst981@134.2.168.52:"/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/$sim_name/plots" "/home/jan/Documents/results_nowcasting/$sim_name"

BEST:
./copy_runs.sh sim_name_1 sim_name_2 ...

#### SCP plots indivisually
scp bst981@134.2.168.72:/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/Run_20231025-102508_ID_4495294several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler/plots/fss/* /home/jan/Documents/results_nowcasting/Run_20231025-102508_ID_4495294several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler_GOOD_PERFORMANCE/plots/fss

### Torch Code

from https://github.com/jthuemmel/SpatioTemporalNetworks/blob/main/train_mixed.py

###
mlflow

activate conda env

start server with, within mlruns folder
mlflow ui --backend-store-uri ./mlruns

go to localhost:5000/


#######Print model parameters:

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

#######Mixed precision training: (16 bit gradients instead of 32 bit):

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

#generate predictions     with autocast():    if leadtime_mode:    leadtime = th.randint(horizon, (1,)).to(device, th.float32)    prediction = model(predictive[:, :, :context], prescribed, leadtime)    target = predictive[:, :, context + leadtime]    else:    prediction = model(predictive[:, :, :context], prescribed, horizon)    target = predictive[:, :, context:]    #calculate loss    loss = latMSE(prediction, target, w)    #backward pass        scaler.scale(loss).backward()    scaler.step(optimizer)           scaler.update()



########Remote interactive session Pycharm:
https://portal.mlcloud.uni-tuebingen.de/user-guide/tutorial/ 

###Installation stuff:
When ssh connection is already set up in pycharm (according to webpage), just create new interpreter and in the process
choose the correct ssh connection. Add interpreter "from existing" and choose path to python3 on remote (ehich python3 when conda env active)

run on slurm:
srun --partition=gpu-v100 --time=0-12:00 --gres=gpu:1 --pty bash
officially:
srun --gres=gpu:1 --pty bash

then run on slurm:
hostname
this will give you the value for num


run on local:

custom:
ssh -AtL 6608:localhost:6608 bst981@134.2.168.72 "ssh -AtL 6608:localhost:22 bst981@slurm-v100-6 bash"

Galvani custom:
ssh -AtL 6608:localhost:6608 bst981@134.2.168.43 "ssh -AtL 6608:localhost:22 bst981@galvani-cn201 bash"


BEST improved
num=1; ssh -AtL 6608:localhost:6608 bst981@134.2.168.72 "ssh -AtL 6608:localhost:22 bst981@slurm-v100-$num bash"

general:
ssh -AtL $B_PORT:localhost:$B_PORT $YOURLOGIN@134.2.168.72 "ssh -AtL $B_PORT:localhost:$COMPUTE_PORT $YOURLOGIN@$NODE bash"



###### Instead of merging git branch, overwrite old branch

$ git checkout new_branch
$ git merge -s ours old_branch # Merge branches, but use our (=staging) branch head
$ git checkout old_branch
$ git merge new_branch

https://stackoverflow.com/questions/4624357/how-do-i-overwrite-rather-than-merge-a-branch-on-another-branch-in-git 


###### Jakob's conda / mamba tipps:

Einfach mamba direkt in $WORK installieren (bash.rc updaten) 

(Conda per default zum mamba solver mit wohl fehleranfällig
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
NICHT EMPFOHLEN!
)

Regelmäßig environments in yml schreiben mit

conda env export --from-history > filename.yml

--from_history stellt sicher, dass nur die manuell installierten Packete drinstehen

Um environment von .yml wiederherzustellen ERST environment erstellen und dann von .yml reininstallieren

mamba create -n phdenv
mamba env update -n phdenv -f phdenv.yml

--> Diese Befehle direkt auf GPU node mit srun (mamba unterstütz GPU parallelisierung)
--> NEIN unterstütz kein GPU!


###### Conda env
Currently working conda env on Cluster:
first_CNN_on_Radolan_3
