
#### Torch module dicts
- They can be used 
Example in https://github.com/jthuemmel/SpatioTemporalNetworks/blob/main/models/models.py
self.decoder_layers = nn.ModuleDict()
self.decoder_layers[f'scale_{i}'](z)

#### Specific stuff for this implementation
Activate remote python venv:
source /home/jan/Programming/remote/first_CNN_on_radolan_remote/virtual_env/bin/activate

For some reason has to be started without sudo!