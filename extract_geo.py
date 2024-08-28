import torch
import numpy as np
import os
import yaml
from scene import Scene, SplatFieldsModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from gaussian_renderer import GaussianModel
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from utils.sh_utils import eval_sh
from utils.general_utils import strip_symmetric, build_scaling_rotation

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    cov_array = torch.stack((
        actual_covariance[:, 0, 0],
        actual_covariance[:, 0, 1],
        actual_covariance[:, 0, 2],
        actual_covariance[:, 1, 1],
        actual_covariance[:, 1, 2],
        actual_covariance[:, 2, 2],
    ), dim=-1)
    return cov_array * 100000

def get_gaussian_dict(_viewpoint_cam, gaussians, deform, iteration=None, warm_up=None, static=False, n_splats=-1):
    fid = _viewpoint_cam.fid.cuda()
    if static:
        fid = 0*fid
    # animate
    if static or (iteration is not None and warm_up is not None and iteration < warm_up):
        gaussian_dict = {
            'means3D': gaussians.get_xyz,
            'active_sh_degree': gaussians.active_sh_degree, # 0
            'gaussian_opacity': gaussians.get_opacity, # N,1 in range [0,1]
            'gaussian_features': gaussians.get_features, # N, 16, 3
            'gaussian_scales': gaussians.get_scaling, # N,3
            'gaussian_rotations': gaussians.get_rotation, # N,4
        }
        shs_view = gaussian_dict['gaussian_features'].transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        dir_pp = gaussian_dict['means3D'] - _viewpoint_cam.camera_center.repeat(gaussian_dict['means3D'].shape[0], 1)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        gaussian_rgb = colors_precomp
        gaussian_dict['gaussian_rgb'] = gaussian_rgb
        overwrite_attributes = None
    else:
        active_sh_degree = gaussians.active_sh_degree
        xyz = gaussians.get_xyz.detach()
        scaling = gaussians.get_scaling.detach()
        gaussian_features = gaussians.get_features
        if n_splats > 0 and n_splats < xyz.shape[0]: # sample splats if needed
            idx = torch.randperm(xyz.shape[0])[:n_splats]
            xyz = xyz[idx]
            scaling = scaling[idx]
            gaussian_features = gaussian_features[idx]
        else:
            idx = None

        N = xyz.shape[0]
        time_input = fid.detach().unsqueeze(0).expand(N, -1)
        ret = deform.step(xyz, time_input)
        gaussian_dict = {
            'idx': idx,
            'means3D': ret['means3D'],
            'active_sh_degree': active_sh_degree, # 0
            'gaussian_opacity': ret['opacity'], # N,1 in range [0,1]
            # 'gaussian_rgb': ret['rgb'], # N, 16, 3
            # 'gaussian_features': ret.get('features', None), # N, 16, 3
            'gaussian_scales': ret['scales'] + scaling, # N,3
            'gaussian_rotations': ret['rotations'], # N,4
            'gradient_error': ret.get('gradient_error', None)
        }
        if 'gaussian_features' in ret:
            gaussian_dict['gaussian_features'] = ret['gaussian_features'].view(gaussian_features.shape)*0.1 + gaussian_features
        elif 'rgb' in ret:
            gaussian_dict['gaussian_rgb'] = ret['rgb']
        elif 'rgb_fnc' in ret:
            gaussian_dict['gaussian_rgb_fnc'] = ret['rgb_fnc']
        else:
            gaussian_dict['gaussian_features'] = gaussian_features
        overwrite_attributes = {}
        overwrite_attributes["xyz"] = gaussian_dict['means3D']
        # gaussian_features = gaussian_dict.get('gaussian_features', None)
        # if gaussian_features is None:
        #     gaussian_features = gaussians.get_features 
        f_dc, f_rest = gaussians.get_features.split([
            gaussians._features_dc.shape[1], 
            gaussians._features_rest.shape[1]
        ], dim=1)
        overwrite_attributes["f_dc"] = f_dc
        overwrite_attributes["f_rest"] = f_rest
        overwrite_attributes["opacity"] = gaussian_dict['gaussian_opacity']
        overwrite_attributes["scaling"] = ret['scales']
        overwrite_attributes["rotation"] = gaussian_dict['gaussian_rotations']

    return gaussian_dict, overwrite_attributes

from pytorch3d.ops.knn import knn_points
def query_nn(pts, n_neighbors=5, eps=1e-5):
    dists, nn_ix, _ = knn_points(pts.unsqueeze(0), pts.unsqueeze(0), K=n_neighbors, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)
    corss_dists = torch.cdist(pts[nn_ix], pts[nn_ix]) # B x N x N
    # convert distances to spatial weights
    weights = torch.full_like(corss_dists, fill_value=eps)
    weights[corss_dists > eps] = 1.0 / corss_dists[corss_dists > eps]
    weights = weights / weights.sum(-1).sum(-1)[:, None, None].clamp_min(1e-5)
    return weights, nn_ix

def morans_measure(weight, feature):
    """
    Args:
        weight: B x N x N matrix of weights
        feature: B x N x F matrix of features
    Returns:
        moran: Moran's I measure
    """
    # skip the first one which corresponds to the centroid
    assert not torch.isnan(weight).any()
    N = feature.shape[1]
    W = weight.sum(-1).sum(-1)[:, None, None] # B x 1 x 1
    w_ij = (N/W)*weight # B x N x N
    assert not torch.isnan(w_ij).any()
    
    # feature_mean = feature.mean(dim=1, keepdim=True) # B x 1 x F
    # x_mean = feature - feature_mean # B x N x F
    x_mean = feature
    denom = (x_mean**2).sum(dim=1) # B x F

    x_mean_bf = x_mean.contiguous().permute(0, 2, 1).reshape(-1, N) # B*F x N
    x_corr = (x_mean_bf.unsqueeze(-1) @ x_mean_bf.unsqueeze(-2)) # B*F x N x N
    x_corr = x_corr.view(x_mean.shape[0], x_mean.shape[2], N, N) # B x F x N x N

    nom = (w_ij.unsqueeze(1) * x_corr).sum(-1).sum(-1) # B x F
    moran = nom / (denom + 1e-4) # B x F

    return moran.mean()

def morans_loss(weight, feature):
    moran_score = morans_measure(weight, feature)
    moran_loss = 1.0 - moran_score.clamp(0, 1)
    return moran_loss

@torch.no_grad()
def render_sets(dataset: ModelParams, hyper: ModelHiddenParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, skip_pred: bool,
                mode: str):
    setattr(hyper, 'n_frames', dataset.load_time_step if dataset.load_time_step > 1 else 0)
    is_static = dataset.is_static
    
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.use_isotropic = hyper.use_isotropic
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    iteration = scene.loaded_iter
    deform = SplatFieldsModel(hyper, dataset.is_blender, radius=scene.cameras_extent)
    deform.load_weights(dataset.model_path, iteration=iteration)

    view = scene.getTestCameras()[0]
    view.load2device()

    gaussian_dict, overwrite_attributes = get_gaussian_dict(view, gaussians, deform, None, None, static=is_static)
    xyz = gaussian_dict['means3D'].detach().cpu().numpy()
    weight_mat, nn_ix = query_nn(gaussian_dict['means3D'])
    NORMALIZE = True
    gaussian_covariance = build_covariance_from_scaling_rotation(gaussian_dict['gaussian_scales'], 1.0, gaussian_dict['gaussian_rotations'])
    gaussian_dict_gaussian_scales = gaussian_dict['gaussian_scales']
    gaussian_dict_gaussian_rotations = gaussian_dict['gaussian_rotations']
    gaussian_dict_gaussian_opacity = gaussian_dict['gaussian_opacity'].clip(0.01, 1)
    gaussian_dict_gaussian_rgb = gaussian_dict['gaussian_rgb']
    gaussian_dict_gaussian_covariance = gaussian_covariance
    if NORMALIZE:
        gaussian_dict_gaussian_scales = (gaussian_dict_gaussian_scales - gaussian_dict_gaussian_scales.mean(0, keepdim=True)) #/ gaussian_dict_gaussian_scales.std(0, keepdim=True)
        gaussian_dict_gaussian_rotations = (gaussian_dict_gaussian_rotations - gaussian_dict_gaussian_rotations.mean(0, keepdim=True)) #/ gaussian_dict_gaussian_rotations.std(0, keepdim=True)
        # gaussian_dict_gaussian_opacity = (gaussian_dict_gaussian_opacity - gaussian_dict_gaussian_opacity.mean(0, keepdim=True)) #/ gaussian_dict_gaussian_opacity.std(0, keepdim=True)
        # gaussian_dict_gaussian_rgb = (gaussian_dict_gaussian_rgb - gaussian_dict_gaussian_rgb.mean(0, keepdim=True)) #/ gaussian_dict_gaussian_rgb.std(0, keepdim=True)
        gaussian_dict_gaussian_covariance = (gaussian_dict_gaussian_covariance - gaussian_dict_gaussian_covariance.mean(0, keepdim=True)) #/ gaussian_dict_gaussian_covariance.std(0, keepdim=True)
    moran_scale = morans_measure(weight_mat, gaussian_dict_gaussian_scales[nn_ix])
    moran_rotation = morans_measure(weight_mat, gaussian_dict_gaussian_rotations[nn_ix])
    moran_opacity = morans_measure(weight_mat, gaussian_dict_gaussian_opacity[nn_ix])
    moran_rgb = morans_measure(weight_mat, gaussian_dict_gaussian_rgb[nn_ix])
    moran_covariance = morans_measure(weight_mat, gaussian_dict_gaussian_covariance[nn_ix])

    print("Moran's I for RGB:", round(moran_rgb.item(), 3))
    print("Moran's I for scale:", round(moran_scale.item(), 3))
    print("Moran's I for rotation:", round(moran_rotation.item(), 3))
    print("Moran's I for opacity:", round(moran_opacity.item(), 3))
    print("Moran's I for covariance:", round(moran_covariance.item(), 3))
    # save moran's I to file
    morons_path = os.path.join(dataset.model_path, f'MoransI_iteration_{iteration}.yaml')
    with open(morons_path, 'w') as file:
        file.write(f'rgb: {round(moran_rgb.item(), 3)}\n')
        file.write(f'scale: {round(moran_scale.item(), 3)}\n')
        file.write(f'rotation: {round(moran_rotation.item(), 3)}\n')
        file.write(f'opacity: {round(moran_opacity.item(), 3)}\n')
        file.write(f'covariance: {moran_covariance.item()}\n')
    print("Saved to", morons_path)
    exit(0)
    # opacities = gaussian_dict['gaussian_opacity'].detach().cpu().numpy()
    # scale = gaussian_dict['gaussian_scales'].detach().cpu().numpy()
    # rotation = gaussian_dict['gaussian_rotations'].detach().cpu().numpy()
    # gaussian_covariance = gaussian_covariance.detach().cpu().numpy()
    # pred_rgb = gaussian_dict['gaussian_rgb'].detach().cpu().numpy()

    # results_path = os.path.join(dataset.model_path, 'vis', "ours_{}.ply".format(iteration))
    results_path = os.path.join(dataset.model_path, 'point_cloud/iteration_70000.ply')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # gaussians.save_ply("point_cloud.ply", overwrite_attributes=overwrite_attributes, vis_geometric=vis_geometric)

    # save the results
    xyz = gaussian_dict['means3D'].detach().cpu().numpy()
    weight_mat, nn_ix = query_nn(gaussian_dict['means3D'])
    pred_rgb = gaussian_dict['gaussian_rgb'].detach().cpu().numpy()
    moran_rgb = morans_measure(weight_mat, gaussian_dict['gaussian_rgb'][nn_ix])
    print("Moran's I for RGB:", moran_rgb.item())

    fused_color = RGB2SH(torch.tensor(np.asarray(pred_rgb)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    f_dc = features[:, :, 0:1].transpose(1, 2).contiguous().flatten(start_dim=1).contiguous().cpu().numpy()*0.0
    f_rest = features[:, :, 1:].transpose(1, 2).contiguous().flatten(start_dim=1).contiguous().cpu().numpy()

    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]

    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    # PlyData([el]).write(results_path)
    gaussians.save_ply(results_path, overwrite_attributes=overwrite_attributes, vis_geometric=True)
    print("Saved to", results_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_pred", action="store_true")
    parser.add_argument("--rnd_depth", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(0,120)] + [100_000, 200_000])
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)

    render_sets(model.extract(args), hp.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_pred, args.mode)
