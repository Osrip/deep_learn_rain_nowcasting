
#### Torch module dicts
- They can be used 
Example in https://github.com/jthuemmel/SpatioTemporalNetworks/blob/main/models/models.py
self.decoder_layers = nn.ModuleDict()
self.decoder_layers[f'scale_{i}'](z)