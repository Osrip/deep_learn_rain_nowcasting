
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

rsync -auvh --info=progress2 --exclude 'venv' --exclude 'runs' --exclude 'dwd_nc' -e ssh $(pwd)/* bst981@134.2.168.52:/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan

SSD direktory to work on!
/mnt/qb/work2/butz1/bst981