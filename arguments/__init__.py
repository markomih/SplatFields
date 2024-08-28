#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs='+')
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs='+')
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.bg_path = ''
        self.is_static = False
        self.vis_geometric = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load_time_step = 100
        self.load_every_nth = 1
        self.pc_path = ''
        self.max_num_pts = -1
        self.n_views = 6
        self.num_pts = 100_000
        self.pts_samples = 'depth'
        self.train_cam_names = ['cam_train_0', 'cam_train_1', 'cam_train_2', 'cam_train_3', 'cam_train_4', 'cam_train_5', 'cam_train_6', 'cam_train_7', 'cam_train_8', 'cam_train_9']
        self.test_cam_names = ['cam_test']
        self.pred_cam_names = ['cam_test']

        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.use_isotropic = False
        self.contract_pts = False
        self.rgb_w = 128
        self.deform_weight = 1.0
        self.D = 8
        self.W = 256
        self.input_ch = 3
        self.multires = 10
        self.num_basis = 4
        self.encoder_type = ''
        self.encoder_args = {}
        self.flow_model = 'offset'
        self.layer_strategy = 'none'
        self.log2_hashmap_size = 20
        self.n_levels = 16
        self.contract_ngp = False
        self.color_model = 'linear' # mlp, hexplane, 'sh'
        self.opacity_model = 'nerf' # nerf or volsdf
        self.opacity_ones = False
        self.use_deform_net = False
        self.opt_pts = False
        self.opt_pts_per_frame = False
        self.encoder_query_scale = 1.0
        self.use_mlp_encoder = False
        self.cat_points = False
        self.dont_cat_time = False
        self.skips = [4]

        # resfields arguments
        self.composition_rank = 10
        self.compression = 'vm'
        # self.mode = 'lookup'
        self.resfield_layers = []

        # version 2 arguments
        self.use_model_v2 = False
        self.geo_model_disable_pts = False
        self.use_view_dep_rgb = False
        
        super().__init__(parser, "ModelHiddenParams")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.n_splats = -1
        self.all_training = False # to render wrt all training views for a randomly sampled time interval
        self.disable_gaussian_opt = False
        self.iterations = 40_000
        self.num_views = 10
        self.warm_up = -1 #300 #3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 45_000
        self.densify_grad_threshold = 0.0002
        self.overwrite_loc = False
        self.lambda_mask = 0.1
        self.lambda_norm = 0.0
        self.lambda_corr = 0.0
        self.lambda_corr_color = 0.0
        self.lambda_norm_mean = 0.0
        self.lambda_depth = 0.0
        self.lambda_opacity = 0.0
        self.lambda_depthl1 = 0.0
        self.lambda_gradient = 0.0
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
