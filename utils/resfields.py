import torch
try:
    import tensorly as tl
    from tensorly.random.base import random_cp
    tl.set_backend('pytorch')
except ImportError:
    pass

class Linear(torch.nn.Linear):
    r"""Applies a ResField Linear transformation to the incoming data: :math:`y = x(A + \delta A_t)^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        rank: value for the the low rank decomposition
        capacity: size of the temporal dimension

    Attributes:
        weight: (F_out x F_in)
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.

    Examples::

        >>> m = nn.Linear(20, 30, rank=10, capacity=100)
        >>> input_x, input_time = torch.randn(128, 20), torch.randn(128)
        >>> output = m(input, input_time)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def minimum(self, x, y):
        return torch.minimum(x, y.repeat(1, x.shape[1]))
    def maximum(self, x, y):
        return torch.maximum(x, y.repeat(1, x.shape[1]))
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, rank=None, capacity=None, mode='lookup', compression='vm', fuse_mode='add', coeff_ratio=1.0, chunk_size=None, chunk_strategy='both', ignore_residuals=False, lock_weights=False, siren_kwargs=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert mode in ['lookup', 'interpolation', 'interpolation_siren', 'cp']
        assert compression in ['vm', 'vm_cum', 'vm_cum_mat', 'cp', 'none', 'none_cum', 'tucker', 'resnet', 'vm_noweight', 'vm_attention', 'loe', 'mm_tensor', 'lora_3', 'lora_ngp']
        assert chunk_strategy in ['shared', 'delta', 'both']
        assert fuse_mode in ['add', 'mul', 'none']
        self.rank = rank
        self.fuse_mode = fuse_mode
        self.capacity = capacity
        self.compression = compression
        self.ignore_residuals = ignore_residuals
        self.lock_weights = lock_weights
        self.mode = mode
        self.fuse_op = {
            'add': torch.add,
            'mul': torch.mul,
            'none': lambda x, y: x
        }[self.fuse_mode]
        self.chunk_size = chunk_size
        self.chunk_strategy = chunk_strategy

        if self.rank is not None and self.capacity is not None and self.capacity > 0:
            n_coefs = int(self.capacity*coeff_ratio)
            if self.compression == 'vm' and chunk_size is not None:
                weights_t = 0.01*torch.randn((n_coefs, self.rank)) # C, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in

                n_chunks = self.capacity // chunk_size
                assert n_chunks > 1, 'chunk_size should be larger than capacity'
                if chunk_strategy in ['shared', 'both']:
                    chunk_weights = 0.01*self.weight.clone().detach()[None].repeat((n_chunks, 1, 1)) # chunk_size, F_out, F_in
                    self.register_parameter('chunk_weights', torch.nn.Parameter(chunk_weights))
                if chunk_strategy in ['delta', 'both']:
                    matrix_t = matrix_t.clone().detach()[None].repeat((n_chunks, 1, 1)) # chunk_size, R, F_out*F_in
                
                self.register_buffer('chunk_size2capacity', torch.arange(n_chunks).repeat_interleave(chunk_size))
                self.register_parameter('weights_t', torch.nn.Parameter(weights_t))
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t))

            elif self.compression == 'vm' or self.compression in ['vm_cum', 'vm_cum_mat']:
                weights_t = 0.01*torch.randn((n_coefs, self.rank)) # C, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                if self.fuse_mode == 'mul': # so that it starts with identity
                    matrix_t.fill_(1.0)
                    weights_t.fill_(1.0/self.rank)
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
                if self.mode == 'interpolation_siren':
                    self.weights_t_siren = SirenMLP(in_features=1, out_features=self.rank, **siren_kwargs)
                else:
                    self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) # C, R
            elif self.compression == 'mm_tensor':
                weights_t = 0.01*torch.randn((n_coefs, self.weight.shape[0], self.rank)) # C, F_out, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[1])) # R, F_in, 1
                self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) 
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) 
            elif self.compression == 'lora_ngp':
                # create ngp
                config = {
                    'encoding': {
                        'otype': 'HashGrid', 
                        'n_levels': 16, 
                        'n_features_per_level': 2, 
                        'log2_hashmap_size': 18, # 2**14 - 2**24
                        'base_resolution': 16, 
                        'per_level_scale': 1.5
                    },
                    'network': {
                        'otype': 'FullyFusedMLP', 
                        'activation': 'ReLU', 
                        'output_activation': 'None', 
                        'n_neurons': 64, 
                        'n_hidden_layers': 1
                    }
                }
                import tinycudann as tcnn
                self.tcnn_coef = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=self.weight.shape[1],
                    encoding_config=config['encoding'],
                    network_config=config['network']
                )
                self.tcnn_bases = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=self.weight.shape[1],
                    encoding_config=config['encoding'],
                    network_config=config['network']
                )


            elif self.compression.startswith('lora'):
                n_dim = int(self.compression.split('_')[1])
                n_ch =  (self.weight.shape[0]+self.weight.shape[1])*self.rank
                shape = [1, n_ch] + [self.capacity]*n_dim  # 1,C,(T),D,H,W
                self.register_parameter('weights_t', torch.nn.Parameter(0.01*torch.randn(shape)))

            elif self.compression == 'loe':
                matrix_t = 0.0*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'vm_attention':
                attention_weight = torch.ones((n_coefs, self.rank)) # C, R
                self.register_parameter('attention_weight', torch.nn.Parameter(attention_weight)) # C, R
                weights_t = 0.01*torch.randn((n_coefs, self.rank)) # C, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                if self.fuse_mode == 'mul': # so that it starts with identity
                    matrix_t.fill_(1.0)
                    weights_t.fill_(1.0/self.rank)
                self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) # C, R
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'vm_noweight':
                matrix_t = 0.000001*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'none' or self.compression == 'none_cum':
                self.register_parameter('matrix_t', torch.nn.Parameter(0.0*torch.randn((self.capacity, self.weight.shape[0]*self.weight.shape[1])))) # C, F_out*F_in
            elif self.compression == 'resnet':
                self.register_parameter('resnet_vec', torch.nn.Parameter(0.0*torch.randn((self.capacity, self.weight.shape[0])))) # C, F_out
            elif self.compression == 'cp':
                weights, factors = random_cp((capacity, self.weight.shape[0], self.weight.shape[1]), self.rank, normalise_factors=False) # F_out, F_in
                self.register_parameter(f'lin_w', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(weights))))
                self.register_parameter(f'lin_f1', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[0]))))
                self.register_parameter(f'lin_f2', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[1]))))
                self.register_parameter(f'lin_f3', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[2]))))
            elif self.compression == 'tucker':
                tmp = tl.decomposition.tucker(self.weight[None].repeat((capacity, 1, 1)), rank=self.rank, init='random', tol=10e-5, random_state=12345, n_iter_max=1)
                self.core = torch.nn.Parameter(0.01*torch.randn_like(tmp.core))
                factors = [0.01*torch.randn_like(_f) for _f in tmp.factors]
                self.factors = torch.nn.ParameterList([torch.nn.Parameter(_f) for _f in factors])
            else:
                raise NotImplementedError

    def query_lora_ngp(self, coords):
        """ lora query
        Args:
            coord: (B, S, 3) noramlized in [-1, 1]
        Returns:
            output: (B, S, F_out)
        """
        assert coords.shape[-1] == 3, 'coord should be 3D'
        coords = coords * 0.5 + 0.5  # rescale to (0, 1)
        if len(coords.shape) == 3:
            B, T, d = coords.shape
        output_coef = self.tcnn_coef(coords.view(-1, coords.shape[-1])).float()
        output_bases = self.tcnn_bases(coords.view(-1, coords.shape[-1])).float()
        if len(coords.shape) == 3:
            output_coef = output_coef.view(B, T, output_coef.shape[-1])
            output_bases = output_bases.view(B, T, output_bases.shape[-1])

        return output_coef, output_bases

    def _get_delta_weight(self, input_time=None, frame_id=None):
        """Returns the delta weight matrix for a given time index.
        
        Args:
            input_time: time index of the input tensor. Data range from -1 to 1. 
                Tensor of shape (N)
        Returns:
            delta weight matrix of shape (N, F_out, F_in)
        """
        # return self.weight + torch.einsum('tr,rfi->tfi', self.weights_t, self.matrix_t)
        weight = self._get_weight()
        if self.compression == 'vm' and self.chunk_size is not None:
            weights_t = self.weights_t # C, R
            matrix_t = self.matrix_t

            if self.chunk_strategy == 'shared':
                # matrix_t.shape = # R, F_out*F_in
                weight = self.chunk_weights[self.chunk_size2capacity] + weight[None]# C, F_out, F_in
                _mat = weights_t @ matrix_t # C,R * R,F_out*F_in -> C,F_out*F_in
    
            elif self.chunk_strategy == 'delta':
                # matrix_t.shape = chunk_size, R, F_out*F_in
                matrix_t = matrix_t[self.chunk_size2capacity] # C, R, F_out*F_in
                _mat = (weights_t.unsqueeze(-1) * matrix_t).sum(-2) # (C,R,1) * (C,R,F_out*F_in) -> C,F_out*F_in
                weight = weight[None] # 1, F_out, F_in
    
            elif self.chunk_strategy == 'both':
                weight = self.chunk_weights[self.chunk_size2capacity] + weight[None]# C, F_out, F_in

                matrix_t = matrix_t[self.chunk_size2capacity] # C, R, F_out*F_in
                _mat = (weights_t.unsqueeze(-1) * matrix_t).sum(-2) # (C,R,1) * (C,R,F_out*F_in) -> C,F_out*F_in
            else:
                raise NotImplementedError
            
            delta_w = self.fuse_op(_mat, weight.view(weight.shape[0], -1)) # C, F_out*F_in
            delta_w = delta_w.t()# F_out*F_in, C
            
        elif self.compression == 'vm':
            if self.mode == 'interpolation':
                grid_query = input_time.view(1, -1, 1, 1) # 1, N, 1, 1

                weights_t = self.weights_t # C, R
                weights_t = torch.nn.functional.grid_sample(
                    weights_t.transpose(0, 1).unsqueeze(0).unsqueeze(-1), # 1, R, C, 1
                    torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1), 
                    padding_mode='border', 
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0).squeeze(-1).transpose(0, 1) # 1, R, N, 1 ->  N, R
            elif self.mode == 'interpolation_siren':
                weights_t = self.weights_t_siren(input_time.view(-1, 1)) # (N,1) -> (N, R)
            else:
                weights_t = self.weights_t # C, R

            delta_w = self.fuse_op((weights_t @ self.matrix_t).t(), weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'vm_cum':
            weights_t = self.weights_t # C, R
            weights_t = torch.cumsum(weights_t, dim=0) # C, R
            delta_w = self.fuse_op((weights_t @ self.matrix_t).t(), weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'mm_tensor':
            weights_t = self.weights_t.view(-1, self.weights_t.shape[-1]) # C*F_out, R
            matrix_t = self.matrix_t  # R, F_in
            _mat = weights_t @ matrix_t # C*F_out, F_in
            _mat = _mat.view(self.weights_t.shape[0], -1) # C, F_out*F_in
            delta_w = self.fuse_op(_mat.transpose(0, 1), weight.view(-1, 1)) # F_out*F_in, C

        elif self.compression == 'vm_cum_mat':
            weights_t = self.weights_t # C, R
            # weights_t = torch.cumsum(weights_t, dim=0) # C, R
            _mat = (weights_t @ self.matrix_t).t() # F_out*F_in, C
            _mat = torch.nn.functional.selu(_mat) # nonlinSelu
            _mat = torch.cumsum(_mat, dim=1)
            delta_w = self.fuse_op(_mat, weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'loe':
            grid_query = input_time.view(1, -1, 1, 1) # 1, N, 1, 1
            delta_w = torch.nn.functional.grid_sample(
                self.matrix_t.transpose(0, 1).unsqueeze(0).unsqueeze(-1), # 1, F_out*F_in, R, 1
                torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1), 
                padding_mode='border', 
                mode='nearest',
                align_corners=True
            ).squeeze(0).squeeze(-1) # 1, F_out*F_in, N, 1 ->  F_out*F_in,N 
        elif self.compression == 'vm_attention':
            attention = torch.softmax(self.attention_weight @ self.attention_weight.t()/self.rank, dim=0) # C,R @ R,C -> C,C
            weights = attention @ self.weights_t # C,C @ C,R -> C,R
            delta_w = self.fuse_op((weights @ self.matrix_t).t(), weight.view(-1, 1))
        elif self.compression == 'vm_noweight':
            delta_w = self.fuse_op(self.matrix_t.t(), weight.view(-1, 1)) # F_out*F_in, C
            delta_w = delta_w.sum(1, keepdim=True).repeat(1, self.capacity) # F_out*F_in, C
        elif self.compression == 'none':
            delta_w = self.fuse_op(self.matrix_t.t(), weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'none_cum':
            _mat = torch.cat((weight.view(-1, 1), self.matrix_t.t()[:, 1:]/250), dim=1)
            #+ self.matrix_t.t()[:, :1],
            delta_w = torch.cumsum(_mat, dim=1) # F_out*F_in, C
        elif self.compression == 'cp':
            _weights = getattr(self, f'lin_w')
            _factors = [getattr(self, f'lin_f1'), getattr(self, f'lin_f2'), getattr(self, f'lin_f3')]
            lin_w = tl.cp_to_tensor((_weights, _factors)) # C, F_out, F_in
            delta_w = self.fuse_op(lin_w.view(lin_w.shape[0], -1).t(), weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'tucker':
            core = getattr(self, f'core')
            factors = getattr(self, f'factors')
            lin_w = tl.tucker_to_tensor((core, factors)) # C, F_out, F_in
            delta_w = self.fuse_op(lin_w.reshape(lin_w.shape[0], -1).t(), weight.view(-1, 1)) # F_out*F_in, C
        else:
            raise NotImplementedError

        mat = delta_w.permute(1, 0).view(-1, *weight.shape)
        if mat.shape[0] == 1:
            out = mat[0]
        else:
            if self.mode in ['interpolation', 'interpolation_siren']:
                out = mat
            else:
                out = mat[frame_id] # N, F_out, F_in

        # if self.mode == 'interpolation':
        #     grid_query = input_time.view(1, -1, 1, 1) # 1, N, 1, 1
        #     out = torch.nn.functional.grid_sample(
        #         delta_w.unsqueeze(0).unsqueeze(-1), # 1, F_out*F_in, C, 1
        #         torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1), 
        #         padding_mode='border', 
        #         mode='bilinear',
        #         align_corners=True
        #     ) # 1, F_out*F_in, N, 1
        #     out = out.view(*self.weight.shape, grid_query.shape[1]).permute(2, 0, 1) # F_out, F_in, N
        # elif self.mode == 'lookup':
        #     out = delta_w.permute(1, 0).view(-1, *self.weight.shape)[frame_id] # N, F_out, F_in
        # else:
        #     raise NotImplementedError

        return out # N, F_out, F_in
    
    @staticmethod
    def feat_sample3d(feat, dhw, mode='bilinear', padding_mode='zeros', align_corners=True):
        '''
        args:
            feat: (B, C, D, H, W)
            dhw: (B, N, 3) [-1, 1]
        return:
            (B, N, C)
        '''
        feat = torch.nn.functional.grid_sample(
            feat,
            dhw.view(dhw.shape[0], -1, 1, 1, dhw.shape[-1]), # B,N,1,1,1,3
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return feat.view(*feat.shape[:2], -1).permute(0, 2, 1)

    def query_lora(self, input, coord):
        """ lora query

        Args:
            input: (B, S, F_in)
            coord: (B, S, 3 or 4) noramlized in [-1, 1]

        Returns:
            output: (B, S, F_out)
        """
        if self.compression == 'lora_3':
            assert coord.shape[-1] == 3, 'coord should be 3D'
            F_out, F_in = self.weight.shape
            weights = self.feat_sample3d(
                self.weights_t, # 1,C,D,H,W
                coord.view(1, -1, coord.shape[-1]), # (1, B*S, 3)
                mode='bilinear',
                padding_mode='border',
                align_corners=True).squeeze(0) # (B*S, C)
            weights_Fout, weights_in = weights.split([self.rank*F_out, self.rank*F_in], dim=-1) # (B*S, F_out*R), (B*S, F_in*R)
            weights_Fout = weights_Fout.view(-1, self.rank, F_out).permute(0, 2, 1) # (B*S, F_out, R)
            weights_in = weights_in.view(-1, self.rank, F_in) # (B*S, R, F_in)

            input_r = weights_in @ input.view(-1, input.shape[-1], 1) # (B*S, R, F_in) @ (B*S, F_in, 1) -> (B*S, R, 1)
            output = weights_Fout @ input_r  # (B*S, F_out, R) @ (B*S, R, 1) -> (B*S, F_out)
            output = output.view(*input.shape[:2], -1) # (B, S, F_out)

        elif self.compression == 'lora_ngp':
            coeff, bases = self.query_lora_ngp(coord) # (B, S, F_in), (B, S, F_out)
            output = (input*coeff).sum(-1, keepdim=True) * bases # (B, S, 1)*(B, S, F_out) -> (B, S, F_out)

        elif self.compression == 'lora_4':
            raise NotImplementedError

        shared_output = torch.nn.functional.linear(input, self.weight, self.bias)
        return shared_output + output

    def _get_weight(self):
        if self.lock_weights:
            return self.weight.clone().detach()
        return self.weight

    def forward(self, input: torch.Tensor, input_time=None, frame_id=None, coordinates=None) -> torch.Tensor:
        """Applies the linear transformation to the incoming data: :math:`y = x(A + \delta A_t)^T + b
        
        Args:
            input: (B, S, F_in)
            input_time: time index of the input tensor. Data range from -1 to 1.
                Tensor of shape (B) or (1)
        Returns:
            output: (B, S, F_out)
        """
        if self.ignore_residuals:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        if self.rank == 0 or self.rank is None or self.capacity == 0 or self.compression == 'resnet':
            return torch.nn.functional.linear(input, self.weight, self.bias)

        if self.compression.startswith('lora') and self.rank > 0 and self.capacity > 0:
            assert coordinates is not None, 'coordinates should be provided for lora'
            return self.query_lora(input, coordinates)

        weight = self._get_delta_weight(input_time, frame_id) # B, F_out, F_in
        if weight.shape[0] == 1 or len(weight.shape) == 2:
            return torch.nn.functional.linear(input, weight.squeeze(0), self.bias)
        else:
            # (B, F_out, F_in) x (B, F_in, S) -> (B, F_out, S)
            if self.bias is not None:
                return (weight @ input.permute(0, 2, 1) + self.bias.view(1, -1, 1)).permute(0, 2, 1) # B, S, F_out
            else:
                return (weight @ input.permute(0, 2, 1)).permute(0, 2, 1)

    def extra_repr(self) -> str:
        _str = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.capacity, self.mode
        )
        if self.rank is not None and self.capacity is not None:
            _str += ', rank={}, capacity={}, compression={}'.format(self.rank, self.capacity, self.compression)
        return _str

import numpy as np
class Sine(torch.nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SirenMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers, out_activation='none'):
        super().__init__()
        dims = [in_features] + [hidden_features for _ in range(num_hidden_layers)] + [out_features]
        self.nl = Sine()
        self.net = []
        for i in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[i], dims[i + 1])
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            self.net.append(lin)
        self.net = torch.nn.Sequential(*self.net)

        self.out_act = {
            'none': lambda x: x,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': torch.relu,
            'selu': torch.selu,
            'softplus': torch.nn.functional.softplus,
            'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
            'elu': torch.nn.functional.elu,
            'leaky_relu': torch.nn.functional.leaky_relu
        }[out_activation]

    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
    
    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, coords):
        x = coords
        for lin in self.net[:-1]:
            x = self.nl(lin(x))
        x = self.net[-1](x)
        return self.out_act(x)
