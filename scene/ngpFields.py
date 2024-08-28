import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()
    
    def setup(self):
        raise NotImplementedError

class NGPMLP(BaseModel):
    def setup(self):
        in_features = self.config.get('in_ch', 3)
        hidden_features = self.config.get('hidden_features', 64)
        self.out_features = self.config.get('out_ch', hidden_features)
        log2_hashmap_size = self.config.get('log2_hashmap_size', 20)
        num_hidden_layers = self.config.get('num_hidden_layers', 2)
        n_levels = self.config.get('n_levels', 16)
        scale = self.config.get('scale', 10.)
        self.register_buffer('inv_scale', torch.tensor(1./scale, dtype=torch.float32))
        radius = self.config.get('radius', None)
        self.contract_ngp = self.config.get('contract_ngp', False)
        if radius is not None:
            self.register_buffer('radius', torch.tensor(radius))
            self.register_buffer('inv_radius', torch.tensor(1.0 / radius))

        config = {
            'encoding': {
                'otype': 'HashGrid', 
                'n_levels': n_levels, 
                'n_features_per_level': 2, 
                'log2_hashmap_size': log2_hashmap_size, # 2**14 - 2**24
                'base_resolution': 16, 
                'per_level_scale': 1.5
                }, 
            'network': {
                'otype': 'FullyFusedMLP', 
                'activation': 'ReLU', 
                'output_activation': 'None', 
                'n_neurons': hidden_features, 
                'n_hidden_layers': num_hidden_layers
            }
        }
        import tinycudann as tcnn
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=in_features,
            n_output_dims=self.out_features,
            encoding_config=config['encoding'],
            network_config=config['network']
        )
    @property
    def out_dim(self):
        return self.out_features

    @staticmethod
    @torch.no_grad()
    def contract_mipnerf360(xyz, roi_min, roi_max):
        xyz_unit = (xyz - roi_min) / (roi_max - roi_min) # roi_to_unit roi -> [0, 1]^3
        xyz_unit = xyz_unit * 2.0 - 1.0 # roi -> [-1, 1]^3
        xyz_norm = torch.norm(xyz_unit, dim=-1, keepdim=True)
        _ind = xyz_norm.squeeze(-1) > 1.0
        xyz_inv_norm = 1./xyz_norm[_ind]
        xyz_unit[_ind] = (2.0 - 1.0 * xyz_inv_norm) * (xyz_unit[_ind] * xyz_inv_norm)
        xyz_unit = xyz_unit * 0.25 + 0.5 # [-1, 1]^3 -> [0.25, 0.75]^3
        return xyz_unit

    def forward(self, coords, frame_id=None, input_time=None):
        # coords: (n_points, dim) # range (-1, 1)
        if self.contract_ngp:
            # map from [-inf,inf] to [0,1]
            coords = self.contract_mipnerf360(coords, -self.radius, self.radius)
        else:
            coords = coords * self.inv_scale # rescale from (-scale, scale) to (-1, 1)
            coords = coords * 0.5 + 0.5  # rescale to (0, 1)
        if len(coords.shape) == 3:
            B, T, d = coords.shape
        output = self.net(coords.half().view(-1, coords.shape[-1])).float()
        if len(coords.shape) == 3:
            output = output.view(B, T, output.shape[-1])
        return output
