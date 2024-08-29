import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .decoders import VAEDecoder
from .time_decoders import TimeVAEDecoder
from mmgen.models import build_module

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()
    
    def setup(self):
        raise NotImplementedError

class LaplaceDensity(BaseModel):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def setup(self):
        beta = self.config.get('beta', 0.1)
        beta_min = self.config.get('beta_min', 0.0001)
        self.register_parameter('beta', torch.nn.Parameter(torch.tensor(beta)))
        self.register_buffer('beta_min', torch.tensor(beta_min))
        
    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

    def inv_s(self):
        return torch.reciprocal(self.get_beta())

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

class BellDensity(BaseModel):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def setup(self):
        # beta = self.config.get('beta', 0.1)
        # beta_min = self.config.get('beta_min', 0.0001)
        # self.register_parameter('beta', torch.nn.Parameter(torch.tensor(beta)))
        # self.register_buffer('beta_min', torch.tensor(beta_min))
        self.register_parameter('lamb', torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('gamma', torch.nn.Parameter(torch.tensor(1.0)))
        
    def inv_s(self):
        return self.lamb

    def forward(self, sdf, beta=None):
        _arg = torch.exp(-self.lamb*sdf)
        return self.gamma*_arg/((1+_arg)**2)
        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

class TriPlaneEncoder(BaseModel):
    def setup(self):
        self.img_resolution = self.config.get('resolution', 200)
        self.img_channels = self.config.get('channels', 16)
        self.fuse_mode = self.config.get('fuse_mode', 'cat')

        self.register_parameter('planes', torch.nn.Parameter(self._triplane_init()))
        self.space_axis = [[0,1], [1,2], [2,0]] # xy, yz, zx

    @property
    def axis(self):
        return self.space_axis

    @property
    def n_planes(self):
        return self.planes.shape[0]

    @property
    def out_dim(self):
        if self.fuse_mode == 'cat':
            return self.n_planes*self.img_channels
        if self.fuse_mode in ['add', 'mean']:
            return self.img_channels
        raise NotImplementedError

    def _fuse_feat(self, feat): # [B, N, n_planes, C]
        if self.fuse_mode == 'cat':
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode in ['add', 'mean']:
            return feat.sum(dim=2)
        raise NotImplementedError

    def _triplane_init(self):
        plane_init = torch.randn([3, self.img_channels, self.img_resolution, self.img_resolution])
        return plane_init # n_planes, C, H, W

    def forward(self, input_pts, alpha_ratio=1.0, input_time=None, frame_id=None):
        # B, N, d = input_pts.shape
        coord = torch.stack([input_pts[..., _ax] for _ax in self.axis]) # [6, B, N, 2]
        sampled_features = F.grid_sample(self.planes, coord) # [6, C, B, N]
        sampled_features = self._fuse_feat(sampled_features.permute(2,3,0,1)) # -> [B, N, F]
        return sampled_features

class GridEncoder(BaseModel):
    def setup(self):
        self.img_resolution = self.config.get('resolution', 128)
        self.img_channels = self.config.get('channels', 24)

        grid = torch.randn([1, self.img_channels, self.img_resolution, self.img_resolution, self.img_resolution])
        self.register_parameter('grid', torch.nn.Parameter(grid))

    @property
    def out_dim(self):
        return self.img_channels

    def forward(self, input_pts, alpha_ratio=1.0, input_time=None, frame_id=None):
        B, N, d = input_pts.shape
        sampled_features = F.grid_sample(self.grid, input_pts[:, None, None])
        sampled_features = sampled_features.view(B, -1, N).permute(0, 2, 1) # B, N, F
        return sampled_features

class VarGridEncoder(GridEncoder):
    def setup(self):
        self.in_ch = self.config.get('in_ch', 8) 
        self.out_ch = self.config.get('out_ch', 16)
        self.noise_res = self.config.get('noise_res', 16) # # grid will be (8->64x64x64)/(16->128x128x128)
        self.tensor_config = self.config.get('tensor_config', ['xy', 'yz', 'zx'])

        self.net = Tensorial3D(self.in_ch, self.out_ch, self.noise_res)

    @property
    def out_dim(self):
        return self.out_ch
    
    @property
    def grid(self):
        return self.net()

class HexPlaneEncoder(TriPlaneEncoder):
    def setup(self):
        super().setup()
        self.time_axis = [[0,3], [1,3], [2,3]] # [xt, yt, zt]

    @property
    def axis(self):
        return self.space_axis + self.time_axis

    def _triplane_init(self):
        planes = super()._triplane_init() # 3, C, H, W
        planes = torch.cat([planes, torch.ones_like(planes)], dim=0) # 6, C, H, W
        return planes

    @property
    def out_dim(self):
        if self.fuse_mode == 'cat':
            return self.n_planes*self.img_channels
        if self.fuse_mode == 'space_cat':
            return 3*self.img_channels
        if self.fuse_mode in ['add', 'mean']:
            return self.img_channels
        raise NotImplementedError

    def _fuse_feat(self, feat): # [B, N, n_planes, C]
        if self.fuse_mode == 'cat':
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode == 'space_cat':
            feat = feat[:, :, 0:3, :] * feat[:, :, 3:, :]
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode in ['add', 'mean']:
            return feat.sum(dim=2)
        raise NotImplementedError

    def forward(self, input_pts, alpha_ratio=1.0, input_time=None, frame_id=None):
        B, N, d = input_pts.shape
        if d == 3 and input_time is not None:
            input_pts = torch.cat([input_pts, input_time.view(1, -1, 1).expand(B, N, 1)*0.8], dim=-1)
        assert input_pts.shape[-1] == 4, 'Input points should be in space-time coordinates'
        return super().forward(input_pts, alpha_ratio, input_time, frame_id)

class Tensorial2D(torch.nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch=8, out_ch=16, noise_res=20, layer_kwargs={}) -> None:
        super().__init__()

        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res)) # 8x20x20
        self.net = build_module(dict(
            # type='VAEDecoder',
            type='TimeVAEDecoder',
            in_channels=noise_ch,
            out_channels=out_ch,
            # up_block_types=('UpDecoderBlock2D',) * 5,
            # block_out_channels=(32, 64, 64, 128, 256),# TODO: OOM
            # up_block_types=('UpDecoderBlock2D',) * 4,
            up_block_types=('TimeUpDecoderBlock2D',) * 4,
            block_out_channels=(32, 32, 32, 32),
            layers_per_block=1,
            layer_kwargs=layer_kwargs,
        ))

    def get_output_shape(self):
        return [self.out_ch, self.noise.size(-2) * self.upx, self.noise.size(-1) * self.upx]

    def forward(self, frame_id):
        # torch.Size([1, 8, 20, 20]) -> torch.Size([1, 16, 320, 320])
        return self.net(self.noise, frame_id=frame_id)

class Decoder1D(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=16,
        upsample_resolutions=(32, 64, 64, 128, 128, 256, 256),
        block_channels=(128, 128, 128, 128, 64, 64, 32, 32)
    ):
        super(Decoder1D, self).__init__()

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=block_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=block_channels[i],
                        out_channels=block_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(16, block_channels[i + 1]),
                    nn.SiLU(),
                )
            )

        self.upsample_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.upsample_layers.append(
                nn.Upsample(
                    size=upsample_resolutions[i],
                    mode="linear",
                    align_corners=False,
                )
            )

        self.conv_out = nn.Conv1d(
            in_channels=block_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.conv_in(x)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.upsample_layers[i](x)

        x = self.conv_out(x)
        x = self.act_fn(x)

        return x

class Tensorial1D(torch.nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res))
        self.net = Decoder1D(
            noise_ch, out_ch,
            tuple(noise_res * i for i in [2, 4, 8, 16, 16]),
            (128, 128, 128, 64, 32, 32)
        )

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx]

    def forward(self):
        return self.net(self.noise)

class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=16,
        upsample_resolutions=(32, 64, 64, 128, 128, 256, 256),
        block_channels=(128, 128, 128, 128, 64, 64, 32, 32)
    ):
        super(Decoder3D, self).__init__()

        self.conv_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=block_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=block_channels[i],
                        out_channels=block_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(16, block_channels[i + 1]),
                    nn.SiLU(),
                )
            )

        self.upsample_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.upsample_layers.append(
                nn.Upsample(
                    size=upsample_resolutions[i],
                    mode="nearest",
                )
            )

        self.conv_out = nn.Conv3d(
            in_channels=block_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.conv_in(x)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.upsample_layers[i](x)

        x = self.conv_out(x)
        x = self.act_fn(x)

        return x

class Tensorial3D(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch=8, out_ch=16, noise_res=4) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res # (8, 16, 4)
        self.upx = 8
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res, noise_res)) # torch.Size([1, 8, 4, 4, 4])
        self.net = Decoder3D(
            noise_ch, out_ch,
            tuple(noise_res * i for i in [1, 1, 2, 4, 8]),
            (128, 128, 128, 64, 32, 32)
        )

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx, self.noise_res * self.upx, self.noise_res * self.upx]

    def forward(self):
        return self.net(self.noise) # [1, 8, 4, 4, 4] -> [1, 16, 32, 32, 32]

class VarTriPlaneEncoder(BaseModel):
    def setup(self):

        self.in_ch = self.config.get('in_ch', 8)
        self.out_ch = self.config.get('out_ch', 16)
        self.noise_res = self.config.get('noise_res', 20)
        self.tensor_config = self.config.get('tensor_config', ['xy', 'yz', 'zx'])
        layer_kwargs = self.config.get('layer_kwargs')

        self.subs = torch.nn.ModuleList([
            Tensorial2D(self.in_ch, self.out_ch, self.noise_res, layer_kwargs=layer_kwargs) 
            for sub in self.tensor_config
        ])

        self.img_channels = self.out_ch
        self.fuse_mode = self.config.get('fuse_mode', 'cat')
        self.space_axis = [[0,1], [1,2], [2,0]] # xy, yz, zx

    def get_planes(self, frame_id):
        ret = []
        for sub in self.subs:
            ret.append(sub(frame_id=frame_id))
        return torch.cat(ret, dim=0)

    @property
    def axis(self):
        return self.space_axis

    @property
    def n_planes(self):
        return len(self.tensor_config)

    @property
    def out_dim(self):
        if self.fuse_mode == 'cat':
            return self.n_planes*self.img_channels
        if self.fuse_mode in ['add', 'mean']:
            return self.img_channels
        raise NotImplementedError

    def _fuse_feat(self, feat): # [B, N, n_planes, C]
        if self.fuse_mode == 'cat':
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode in ['add', 'mean']:
            return feat.sum(dim=2)
        raise NotImplementedError

    def forward(self, input_pts, alpha_ratio=1.0, input_time=None, frame_id=None):
        # B, N, d = input_pts.shape
        planes = self.get_planes(frame_id=frame_id)
        coord = torch.stack([input_pts[..., _ax] for _ax in self.axis]) # [6, B, N, 2]
        sampled_features = F.grid_sample(planes, coord) # [6, C, B, N]
        sampled_features = self._fuse_feat(sampled_features.permute(2,3,0,1)) # -> [B, N, F]
        return sampled_features

class VarHexPlaneEncoder(VarTriPlaneEncoder):
    def setup(self):
        self.config['tensor_config'] =  ['xy', 'yz', 'zx', 'xt', 'yt', 'zt']
        super().setup()
        self.time_axis = [[0,3], [1,3], [2,3]] # [xt, yt, zt]

    @property
    def axis(self):
        return self.space_axis + self.time_axis

    @property
    def out_dim(self):
        if self.fuse_mode == 'cat':
            return self.n_planes*self.img_channels
        if self.fuse_mode == 'space_cat':
            return 3*self.img_channels
        if self.fuse_mode in ['add', 'mean']:
            return self.img_channels
        raise NotImplementedError

    def _fuse_feat(self, feat): # [B, N, n_planes, C]
        if self.fuse_mode == 'cat':
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode == 'space_cat':
            feat = feat[:, :, 0:3, :] * feat[:, :, 3:, :]
            return feat.reshape(feat.shape[0], feat.shape[1], -1)
        if self.fuse_mode in ['add', 'mean']:
            return feat.sum(dim=2)
        raise NotImplementedError

    def forward(self, input_pts, alpha_ratio=1.0, input_time=None, frame_id=None):
        B, N, d = input_pts.shape
        if d == 3 and input_time is not None:
            input_pts = torch.cat([input_pts, input_time.view(1, -1, 1).expand(B, N, 1)*0.8], dim=-1)
        assert input_pts.shape[-1] == 4, 'Input points should be in space-time coordinates'
        return super().forward(input_pts, alpha_ratio, input_time, frame_id)


# @MODULES.register_module()
# class VAEDecoder(Decoder, ModelMixin):
#     def __init__(
#             self,
#             in_channels=12,
#             out_channels=24,
#             up_block_types=('UpDecoderBlock2D',),
#             block_out_channels=(64,),
#             layers_per_block=2,
#             norm_num_groups=32,
#             act_fn='silu',
#             norm_type='group',
#             zero_init_residual=True):
#         super(VAEDecoder, self).__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             up_block_types=up_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=norm_num_groups,
#             act_fn=act_fn,
#             norm_type=norm_type)
#         self.zero_init_residual = zero_init_residual
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, nn.GroupNorm):
#                 constant_init(m, 1)

#         if self.zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, ResnetBlock2D):
#                     constant_init(m.conv2, 0)
#                 elif isinstance(m, Attention):
#                     constant_init(m.to_out[0], 0)

# @MODULES.register_module()
# class TimeVAEDecoder(Decoder, ModelMixin):
#     def __init__(
#             self,
#             in_channels=12,
#             out_channels=24,
#             up_block_types=('UpDecoderBlock2D',),
#             block_out_channels=(64,),
#             layers_per_block=2,
#             norm_num_groups=32,
#             act_fn='silu',
#             norm_type='group',
#             zero_init_residual=True):
#         super(TimeVAEDecoder, self).__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             up_block_types=up_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=norm_num_groups,
#             act_fn=act_fn,
#             norm_type=norm_type)
#         self.zero_init_residual = zero_init_residual
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, nn.GroupNorm):
#                 constant_init(m, 1)

#         if self.zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, ResnetBlock2D):
#                     constant_init(m.conv2, 0)
#                 elif isinstance(m, Attention):
#                     constant_init(m.to_out[0], 0)

# from diffusers.models.vae import Upsample2D, UNetMidBlock2D, SpatialNorm, is_torch_version

# class TimeUNetMidBlock2D(UNetMidBlock2D):
#     def __init__(self, *input, **kwargs):
#         super().__init__(*input, **kwargs)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             pass # todo add dep params


# class TimeDecoder(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=3,
#         up_block_types=("UpDecoderBlock2D",),
#         block_out_channels=(64,),
#         layers_per_block=2,
#         norm_num_groups=32,
#         act_fn="silu",
#         norm_type="group",  # group, spatial
#     ):
#         super().__init__()
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[-1],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])

#         temb_channels = in_channels if norm_type == "spatial" else None

#         # mid
#         self.mid_block = TimeUNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#             temb_channels=temb_channels,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=self.layers_per_block + 1,
#                 in_channels=prev_output_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=None,
#                 add_upsample=not is_final_block,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attention_head_dim=output_channel,
#                 temb_channels=temb_channels,
#                 resnet_time_scale_shift=norm_type,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         # out
#         if norm_type == "spatial":
#             self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
#         else:
#             self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

#         self.gradient_checkpointing = False

#     def forward(self, z, latent_embeds=None):
#         sample = z
#         sample = self.conv_in(sample)

#         upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
#         if self.training and self.gradient_checkpointing:

#             def create_custom_forward(module):
#                 def custom_forward(*inputs):
#                     return module(*inputs)

#                 return custom_forward

#             if is_torch_version(">=", "1.11.0"):
#                 # middle
#                 sample = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
#                 )
#                 sample = sample.to(upscale_dtype)

#                 # up
#                 for up_block in self.up_blocks:
#                     sample = torch.utils.checkpoint.checkpoint(
#                         create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
#                     )
#             else:
#                 # middle
#                 sample = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(self.mid_block), sample, latent_embeds
#                 )
#                 sample = sample.to(upscale_dtype)

#                 # up
#                 for up_block in self.up_blocks:
#                     sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
#         else:
#             # middle
#             sample = self.mid_block(sample, latent_embeds)
#             sample = sample.to(upscale_dtype)

#             # up
#             for up_block in self.up_blocks:
#                 sample = up_block(sample, latent_embeds)

#         # post-process
#         if latent_embeds is None:
#             sample = self.conv_norm_out(sample)
#         else:
#             sample = self.conv_norm_out(sample, latent_embeds)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample

# def get_up_block(
#     up_block_type,
#     num_layers,
#     in_channels,
#     out_channels,
#     prev_output_channel,
#     temb_channels,
#     add_upsample,
#     resnet_eps,
#     resnet_act_fn,
#     transformer_layers_per_block=1,
#     num_attention_heads=None,
#     resnet_groups=None,
#     cross_attention_dim=None,
#     dual_cross_attention=False,
#     use_linear_projection=False,
#     only_cross_attention=False,
#     upcast_attention=False,
#     resnet_time_scale_shift="default",
#     attention_type="default",
#     resnet_skip_time_act=False,
#     resnet_out_scale_factor=1.0,
#     cross_attention_norm=None,
#     attention_head_dim=None,
#     upsample_type=None,
#     dropout=0.0,
# ):
#     if up_block_type == "UpDecoderBlock2D":
#         return UpDecoderBlock2D(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             dropout=dropout,
#             add_upsample=add_upsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#             temb_channels=temb_channels,
#         )
#     elif up_block_type == "TimeUpDecoderBlock2D":
#         return TimeUpDecoderBlock2D(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             dropout=dropout,
#             add_upsample=add_upsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#             temb_channels=temb_channels,
#         )
#     raise NotImplementedError

# class TimeVAEDecoder(TimeDecoder, ModelMixin):
#     def __init__(
#             self,
#             in_channels=12,
#             out_channels=24,
#             up_block_types=('UpDecoderBlock2D',),
#             block_out_channels=(64,),
#             layers_per_block=2,
#             norm_num_groups=32,
#             act_fn='silu',
#             norm_type='group',
#             zero_init_residual=True):
#         super(TimeVAEDecoder, self).__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             up_block_types=up_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=norm_num_groups,
#             act_fn=act_fn,
#             norm_type=norm_type)
#         self.zero_init_residual = zero_init_residual
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, nn.GroupNorm):
#                 constant_init(m, 1)

#         if self.zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, ResnetBlock2D):
#                     constant_init(m.conv2, 0)
#                 elif isinstance(m, Attention):
#                     constant_init(m.to_out[0], 0)

# class UpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#     ):
#         super().__init__()
#         resnets = []

#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#         self.resnets = nn.ModuleList(resnets)

#         if add_upsample:
#             self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None

#     def forward(self, hidden_states, temb=None, scale: float = 1.0):
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states, temb=temb, scale=scale)

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states)

#         return hidden_states

# class TimeUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#     ):
#         super().__init__()
#         resnets = []

#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#         self.resnets = nn.ModuleList(resnets)

#         if add_upsample:
#             self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None

#     def forward(self, hidden_states, temb=None, scale: float = 1.0):
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states, temb=temb, scale=scale)

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states)

#         return hidden_states


# class TimeResnetBlock2D(nn.Module):
#     r"""
#     A Resnet block.

#     Parameters:
#         in_channels (`int`): The number of channels in the input.
#         out_channels (`int`, *optional*, default to be `None`):
#             The number of output channels for the first conv2d layer. If None, same as `in_channels`.
#         dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
#         temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
#         groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
#         groups_out (`int`, *optional*, default to None):
#             The number of groups to use for the second normalization layer. if set to None, same as `groups`.
#         eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
#         non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
#         time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
#             By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
#             "ada_group" for a stronger conditioning with scale and shift.
#         kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
#             [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
#         output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
#         use_in_shortcut (`bool`, *optional*, default to `True`):
#             If `True`, add a 1x1 nn.conv2d layer for skip-connection.
#         up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
#         down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
#         conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
#             `conv_shortcut` output.
#         conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
#             If None, same as `out_channels`.
#     """

#     def __init__(
#         self,
#         *,
#         in_channels,
#         out_channels=None,
#         conv_shortcut=False,
#         dropout=0.0,
#         temb_channels=512,
#         groups=32,
#         groups_out=None,
#         pre_norm=True,
#         eps=1e-6,
#         non_linearity="swish",
#         skip_time_act=False,
#         time_embedding_norm="default",  # default, scale_shift, ada_group, spatial
#         kernel=None,
#         output_scale_factor=1.0,
#         use_in_shortcut=None,
#         up=False,
#         down=False,
#         conv_shortcut_bias: bool = True,
#         conv_2d_out_channels: Optional[int] = None,
#     ):
#         super().__init__()
#         self.pre_norm = pre_norm
#         self.pre_norm = True
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut
#         self.up = up
#         self.down = down
#         self.output_scale_factor = output_scale_factor
#         self.time_embedding_norm = time_embedding_norm
#         self.skip_time_act = skip_time_act

#         if groups_out is None:
#             groups_out = groups

#         if self.time_embedding_norm == "ada_group":
#             self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
#         elif self.time_embedding_norm == "spatial":
#             self.norm1 = SpatialNorm(in_channels, temb_channels)
#         else:
#             self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

#         self.conv1 = LoRACompatibleConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         if temb_channels is not None:
#             if self.time_embedding_norm == "default":
#                 self.time_emb_proj = LoRACompatibleLinear(temb_channels, out_channels)
#             elif self.time_embedding_norm == "scale_shift":
#                 self.time_emb_proj = LoRACompatibleLinear(temb_channels, 2 * out_channels)
#             elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
#                 self.time_emb_proj = None
#             else:
#                 raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
#         else:
#             self.time_emb_proj = None

#         if self.time_embedding_norm == "ada_group":
#             self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
#         elif self.time_embedding_norm == "spatial":
#             self.norm2 = SpatialNorm(out_channels, temb_channels)
#         else:
#             self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

#         self.dropout = torch.nn.Dropout(dropout)
#         conv_2d_out_channels = conv_2d_out_channels or out_channels
#         self.conv2 = LoRACompatibleConv(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

#         self.nonlinearity = get_activation(non_linearity)

#         self.upsample = self.downsample = None
#         if self.up:
#             if kernel == "fir":
#                 fir_kernel = (1, 3, 3, 1)
#                 self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
#             elif kernel == "sde_vp":
#                 self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
#             else:
#                 self.upsample = Upsample2D(in_channels, use_conv=False)
#         elif self.down:
#             if kernel == "fir":
#                 fir_kernel = (1, 3, 3, 1)
#                 self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
#             elif kernel == "sde_vp":
#                 self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
#             else:
#                 self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

#         self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

#         self.conv_shortcut = None
#         if self.use_in_shortcut:
#             self.conv_shortcut = LoRACompatibleConv(
#                 in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
#             )

#     def forward(self, input_tensor, temb, scale: float = 1.0):
#         hidden_states = input_tensor

#         if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
#             hidden_states = self.norm1(hidden_states, temb)
#         else:
#             hidden_states = self.norm1(hidden_states)

#         hidden_states = self.nonlinearity(hidden_states)

#         if self.upsample is not None:
#             # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
#             if hidden_states.shape[0] >= 64:
#                 input_tensor = input_tensor.contiguous()
#                 hidden_states = hidden_states.contiguous()
#             input_tensor = (
#                 self.upsample(input_tensor, scale=scale)
#                 if isinstance(self.upsample, Upsample2D)
#                 else self.upsample(input_tensor)
#             )
#             hidden_states = (
#                 self.upsample(hidden_states, scale=scale)
#                 if isinstance(self.upsample, Upsample2D)
#                 else self.upsample(hidden_states)
#             )
#         elif self.downsample is not None:
#             input_tensor = (
#                 self.downsample(input_tensor, scale=scale)
#                 if isinstance(self.downsample, Downsample2D)
#                 else self.downsample(input_tensor)
#             )
#             hidden_states = (
#                 self.downsample(hidden_states, scale=scale)
#                 if isinstance(self.downsample, Downsample2D)
#                 else self.downsample(hidden_states)
#             )

#         hidden_states = self.conv1(hidden_states, scale)

#         if self.time_emb_proj is not None:
#             if not self.skip_time_act:
#                 temb = self.nonlinearity(temb)
#             temb = self.time_emb_proj(temb, scale)[:, :, None, None]

#         if temb is not None and self.time_embedding_norm == "default":
#             hidden_states = hidden_states + temb

#         if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
#             hidden_states = self.norm2(hidden_states, temb)
#         else:
#             hidden_states = self.norm2(hidden_states)

#         if temb is not None and self.time_embedding_norm == "scale_shift":
#             scale, shift = torch.chunk(temb, 2, dim=1)
#             hidden_states = hidden_states * (1 + scale) + shift

#         hidden_states = self.nonlinearity(hidden_states)

#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.conv2(hidden_states, scale)

#         if self.conv_shortcut is not None:
#             input_tensor = self.conv_shortcut(input_tensor, scale)

#         output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

#         return output_tensor

