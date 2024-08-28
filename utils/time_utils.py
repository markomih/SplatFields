import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3, scaled_exp_se3, from_homogenous, to_homogenous
from utils import resfields
from scene.tripFields import TriPlaneEncoder, VarTriPlaneEncoder, HexPlaneEncoder, VarHexPlaneEncoder, GridEncoder, VarGridEncoder, LaplaceDensity, BellDensity

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def init_dct_basis(num_basis, num_frames):
    """ Initialize motion basis with DCT coefficient. """
    T = num_frames
    K = num_basis
    dct_basis = torch.zeros([T, K])

    for t in range(T):
        for k in range(1, K + 1):
          dct_basis[t, k - 1] = np.sqrt(2.0 / T) * np.cos(np.pi / (2.0 * T) * (2 * t + 1) * k)
    return dct_basis

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

class GeneralMLP(torch.nn.Module):
    def __init__(self, 
                 in_features=3, 
                 out_features=3, 
                 hidden_features=128, 
                 num_hidden_layers=8, 
                 skips=[4],
                 multires=6, 
                 out_activation='none', act='relu', 
                 composition_rank=0, n_frames=100
                 ):
        super().__init__()
        self.out_features = out_features
        dims = [in_features] + [hidden_features for _ in range(num_hidden_layers)] + [out_features]
        resfield_layers = range(len(dims))[1:-1]

        self.input_ch = in_features
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        in_features = in_features - 3 + xyz_input_ch
        
        self.skips = skips
        
        def _create_lin(layer_id):
            _rank = composition_rank if layer_id in resfield_layers else 0
            _capacity = n_frames if layer_id in resfield_layers and composition_rank > 0 else 0

            if layer_id is None:
                _in, _out = in_features, hidden_features
                lin = resfields.Linear(in_features, hidden_features)
            else:
                _in, _out = (hidden_features, hidden_features) if layer_id not in self.skips else (hidden_features + in_features, hidden_features)
            lin = resfields.Linear(_in, _out, rank=_rank, capacity=_capacity)
            return lin

        self.net = nn.ModuleList(
            [resfields.Linear(in_features, hidden_features)] +
            [_create_lin(i) for i in range(num_hidden_layers)] +
            [resfields.Linear(hidden_features, out_features)]
        )

        _activatinos = {
            'none': lambda x: x,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': torch.relu,
            'selu': torch.selu,
            'softplus': torch.nn.functional.softplus,
            'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
            'elu': torch.nn.functional.elu,
            'normalize': torch.nn.functional.normalize,
            'leaky_relu': torch.nn.functional.leaky_relu
        }
        self.out_act = _activatinos[out_activation]
        self.act = _activatinos[act]

    def forward(self, xyz, xyz_feat=None, frame_id=None):
        xyz_emb = self.embed_fn(xyz)
        h_in = xyz_emb
        if xyz_feat is not None:
            h_in = torch.cat([h_in, xyz_feat], dim=-1)

        h = h_in
        for i, layer in enumerate(self.net):
            h = self.act(layer(h, frame_id=frame_id))
            if i in self.skips and i != len(self.net)-1:
                h = torch.cat([h_in, h], -1)

        out = self.out_act(h)
        return out


class FlowHead(nn.Module):
    def __init__(self, W=256, flow_model='offset', num_basis=4, n_frames=100):
        super(FlowHead, self).__init__()
        self.W = W
        self.flow_model = flow_model
        self.n_frames = n_frames
        self.num_basis = num_basis

        if flow_model == 'offset':
            self.gaussian_warp = nn.Linear(W, 3)
        elif flow_model == 'se3':
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        elif flow_model == 'se3Affine':
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
            self.branch_offset = nn.Linear(W, 3)
        elif flow_model == 'se3Scaled':
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
            self.branch_scale = nn.Linear(W, 1)
            self.branch_offset = nn.Linear(W, 3)
        elif flow_model in ['affine']:
            self.branch_w = nn.Linear(W, 3*3)
            self.branch_v = nn.Linear(W, 3)
        elif self.flow_model in ['dct', 'dct_siren']:
            self.num_basis = num_basis
            self.branch_coeff = nn.Linear(W, out_features = 3 * self.num_basis)
            self.branch_coeff.weight.data.fill_(0.0)
            self.branch_coeff.bias.data.fill_(0.0)
            if self.flow_model == 'dct':
                dct_basis = init_dct_basis(self.num_basis, self.n_frames*2)
                self.register_parameter('trajectory_basis', torch.nn.Parameter(dct_basis)) # n_frames, n_bases
            elif self.flow_model == 'dct_siren':
                self.basis_net = SirenMLP(
                    in_features=1,
                    out_features=self.num_basis,
                    hidden_features=128,
                    num_hidden_layers=2,
                    out_activation='none'
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
        
    def forward(self, hidden, pts, time_step=None, frame_id=None):
        if self.flow_model == 'offset':
            flow = self.gaussian_warp(hidden)

            means3D = pts + flow

        elif self.flow_model == 'se3':
            w = self.branch_w(hidden)
            v = self.branch_v(hidden)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            flow = exp_se3(screw_axis, theta)

            means3D = from_homogenous(torch.bmm(flow, to_homogenous(pts).unsqueeze(-1)).squeeze(-1))
        elif self.flow_model in ['affine']:
            v = self.branch_v(hidden)
            affine = self.branch_w(hidden)
            affine = affine.view(*affine.shape[:-1], 3, 3)
            means3D = torch.bmm(affine, pts.unsqueeze(-1)).squeeze(-1) + v
            flow = means3D - pts
        elif self.flow_model in ['se3Affine']:
            w = self.branch_w(hidden)
            v = self.branch_v(hidden)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            flow = exp_se3(screw_axis, theta)

            _flow = self.branch_offset(hidden)
            means3D = from_homogenous(torch.bmm(flow, to_homogenous(pts).unsqueeze(-1)).squeeze(-1))+_flow
            flow = means3D - pts
        elif self.flow_model in ['se3Scaled']:
            _scale = F.softplus(self.branch_scale(hidden))
            w = self.branch_w(hidden)
            v = self.branch_v(hidden)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            transform_mat = scaled_exp_se3(screw_axis, theta, scale=_scale)

            _offset = self.branch_offset(hidden)
            means3D = from_homogenous(torch.bmm(transform_mat, to_homogenous(pts).unsqueeze(-1)).squeeze(-1))+_offset
            flow = means3D - pts

        elif self.flow_model in ['dct', 'dct_siren']:
            coeff = self.branch_coeff(hidden) # [N, 3 * num_basis]
            coeff = coeff.view(-1, 3, self.num_basis) # [N, 3, num_basis]
            if self.flow_model == 'dct':
                bases = self.trajectory_basis[frame_id.long()] # [num_basis,]
            elif self.flow_model == 'dct_siren':
                bases = self.basis_net(time_step.view(1, 1))
            else:
                raise NotImplementedError
            flow = (coeff*bases.view(1, 1, -1)).sum(-1) # [N, 3]
            means3D = pts + flow

        else:
            raise NotImplementedError

        return flow, means3D

class SplatFields(nn.Module):
    def __init__(self, radius=None, n_frames=0, **kwargs):
        super(SplatFields, self).__init__()

        # shortcuts
        composition_rank = kwargs.get('composition_rank', 0)
        self.n_frames = n_frames

        # create feature extractor encoder
        self.encoder_type = kwargs.get('encoder_type', 'VarTriPlaneEncoder')
        if self.encoder_type in ['VarTriPlaneEncoder']:
            encoder_args = kwargs.get('encoder_args', {})
            encoder_args['layer_kwargs'] = {
                'n_frames': self.n_frames,
                'strategy': kwargs.get('layer_strategy', 'none'),
            }
            encoder_args.update({
                'log2_hashmap_size': kwargs.get('log2_hashmap_size', 20),
                'n_levels': kwargs.get('n_levels', 16),
                'radius': radius,
                'contract_ngp': kwargs.get('contract_ngp', False),
            })
            self.encoder = eval(self.encoder_type)(encoder_args)
            self.feat_dim = self.encoder.out_dim
            self.mlp_refine_feat = torch.nn.Sequential(
                torch.nn.Linear(self.feat_dim, self.feat_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.feat_dim, self.feat_dim),
            )
        else:
            self.feat_dim = 0
        
        # create time encoder
        if n_frames > 0:
            time_multires = kwargs.get('time_multires', 3)
            self.embed_time_fn, time_input_ch = get_embedder(time_multires, 1)
        else:
            time_input_ch = 0

        # create MLP deform network
        self.deform_weight = kwargs.get('deform_weight', 1.0) # set 0 for front-facing scenes
        self.mlp_deform = GeneralMLP(
            in_features=3+self.feat_dim+time_input_ch, 
            out_features=3, 
            hidden_features=kwargs.get('deform_w', 128), 
            num_hidden_layers=kwargs.get('deform_d', 6), 
            skips=kwargs.get('deform_skips', [3]),
            multires=kwargs.get('deform_multires', 6), 
            out_activation='none', 
            act='leaky_relu', 
            composition_rank=composition_rank, n_frames=n_frames
        )

        # create color model
        self.use_view_dep_rgb = kwargs.get('use_view_dep_rgb', False)
        self.mlp_rgb = GeneralMLP(
            in_features=3+self.feat_dim+time_input_ch, 
            out_features=kwargs.get('rgb_w', 128) if self.use_view_dep_rgb else 3, 
            hidden_features=kwargs.get('rgb_w', 128), 
            num_hidden_layers=kwargs.get('rgb_d', 6), 
            skips=kwargs.get('rgb_skips', [3]),
            multires=kwargs.get('rgb_multires', 6), 
            out_activation='none' if self.use_view_dep_rgb else 'sigmoid', 
            act='leaky_relu', 
            composition_rank=composition_rank, n_frames=n_frames
        )
        if self.use_view_dep_rgb:
            self.mlp_rgb_viewdep = torch.nn.Sequential(
                torch.nn.Linear(3+self.mlp_rgb.out_features, 3),
                torch.nn.Sigmoid()
            )
        # create scale model
            
        self.geo_model_disable_pts = bool(kwargs.get('geo_model_disable_pts', False))
        geo_in_feat = 3+self.feat_dim+time_input_ch
        if self.geo_model_disable_pts:
            geo_in_feat -= 3
        self.mlp_scale = GeneralMLP(
            in_features=geo_in_feat, 
            out_features=3, 
            hidden_features=kwargs.get('scale_w', 64), 
            num_hidden_layers=kwargs.get('scale_d', 4), 
            skips=kwargs.get('scale_skips', [2]),
            multires=kwargs.get('scale_multires', 4) if not self.geo_model_disable_pts else 0, 
            out_activation='none', 
            act='leaky_relu', 
            composition_rank=composition_rank, n_frames=n_frames
        )
        # create opacity model
        self.mlp_opacity = GeneralMLP(
            in_features=geo_in_feat, 
            out_features=1, 
            hidden_features=kwargs.get('opacity_w', 64), 
            num_hidden_layers=kwargs.get('opacity_d', 4), 
            skips=kwargs.get('opacity_skips', [2]),
            multires=kwargs.get('opacity_multires', 3) if not self.geo_model_disable_pts else 0, 
            out_activation='sigmoid', 
            act='leaky_relu', 
            composition_rank=composition_rank, n_frames=n_frames
        )
        # create rotation model
        self.mlp_rotation = GeneralMLP(
            in_features=geo_in_feat, 
            out_features=4, 
            hidden_features=kwargs.get('rotation_w', 64), 
            num_hidden_layers=kwargs.get('rotation_d', 3), 
            skips=kwargs.get('rotation_skips', [20]),
            multires=kwargs.get('rotation_multires', 3) if not self.geo_model_disable_pts else 0, 
            out_activation='normalize',
            act='leaky_relu', 
            composition_rank=composition_rank, n_frames=n_frames
        )
        
        # create flow model for dynamic scenes
        if n_frames > 0:
            self.mlp_flow = GeneralMLP(
                in_features=3+self.feat_dim+time_input_ch,
                out_features=kwargs.get('flow_w', 128),
                hidden_features=kwargs.get('flow_w', 128),
                num_hidden_layers=kwargs.get('flow_d', 6),
                skips=kwargs.get('flow_skips', [3]),
                multires=kwargs.get('flow_multires', 6),
                out_activation='none',
                act='leaky_relu',
                composition_rank=composition_rank, n_frames=n_frames
            )
            self.mlp_flow_head = FlowHead(
                W=self.mlp_flow.out_features,
                flow_model=kwargs.get('flow_model', 'se3'),
                num_basis=kwargs.get('dct_basis', 4),
                n_frames=n_frames
            )

        print(self)

    def _time2frame_id(self, t):
        frame_id = t*(self.n_frames-1)
        return torch.round(frame_id)

    def log_variables(self):
        to_ret = {}
        return to_ret
    
    def extract_features(self, x, t):
        t_feat = self.embed_time_fn(t) if self.n_frames > 0 else None
        x_feat = self.encoder(x[None]).squeeze(0) if self.feat_dim > 0 else None
        if x_feat is not None:
            x_feat = self.mlp_refine_feat(x_feat)

        # concatenate features if they are not None
        if t_feat is not None and x_feat is not None:
            h = torch.cat([x_feat, t_feat], dim=-1)
        elif t_feat is not None:
            h = t_feat
        elif x_feat is not None:
            h = x_feat
        else:
            h = None
        return h


    def forward(self, xyz_in, t):
        to_ret = {}

        # parse time step and frame id
        time_step, frame_id = None, None
        if self.n_frames > 0:
            time_step = t.view(-1)[0]
            frame_id = self._time2frame_id(time_step).long()

        # 1) encode points
        pts_feat = self.extract_features(xyz_in, t)

        # 2) move points from optimization to the desired location
        if self.deform_weight > 0:
            xyz_can = xyz_in + self.deform_weight*self.mlp_deform(xyz=xyz_in, xyz_feat=pts_feat, frame_id=frame_id)
        else:
            xyz_can = xyz_in

        # 3) compute scale, opacity, rotation
        geo_xyz, geo_feat = xyz_can, pts_feat
        if self.geo_model_disable_pts:
            geo_xyz, geo_feat = pts_feat, None
        to_ret['scales'] = self.mlp_scale(xyz=geo_xyz, xyz_feat=geo_feat, frame_id=frame_id)
        to_ret['opacity'] = self.mlp_opacity(xyz=geo_xyz, xyz_feat=geo_feat, frame_id=frame_id)
        to_ret['rotations'] = self.mlp_rotation(xyz=geo_xyz, xyz_feat=geo_feat, frame_id=frame_id)

        # 4) compute color 
        rgb = self.mlp_rgb(xyz=xyz_can, xyz_feat=pts_feat, frame_id=frame_id)
        if self.use_view_dep_rgb:
            to_ret['rgb_fnc'] = lambda viewdir: self.mlp_rgb_viewdep(torch.cat([rgb, viewdir], dim=-1))
        else:
            to_ret['rgb'] = rgb

        # 5) compute flow
        if self.n_frames > 0:
            flow_feat = self.mlp_flow(xyz=xyz_can, xyz_feat=pts_feat, frame_id=frame_id)
            flow, means3D = self.mlp_flow_head(hidden=flow_feat, pts=xyz_can, time_step=time_step, frame_id=frame_id)
        else:
            flow, means3D = None, xyz_can

        to_ret.update({ 'flow': flow, 'means3D': means3D, })
        return to_ret
