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

import torch
from scene import Scene, SplatFieldsModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from train import get_gaussian_dict
import glob
import collections
import math
import cv2
from typing import Optional
from scipy import signal
import lpips

def compute_psnr(img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / math.log(10)*torch.log(mse.mean())

def compute_ssim(
    # img0: jnp.ndarray,
    img0: torch.Tensor,
    # img1: jnp.ndarray,
    img1: torch.Tensor,
    # mask: Optional[jnp.ndarray] = None,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
# ) -> jnp.ndarray:
) -> torch.Tensor:
    """Computes SSIM between two images.

    This function was modeled after tf.image.ssim, and should produce
    comparable output.

    Image Inpainting for Irregular Holes Using Partial Convolutions.
        Liu et al., ECCV 2018.
        https://arxiv.org/abs/1804.07723

    Note that the mask operation is implemented as partial convolution. See
    Section 3.1.

    Args:
        img0 (jnp.ndarray): An image of size (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of size (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.
        max_val (float): The dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size (int): Size of the Gaussian blur kernel used to smooth the
            input images.
        filter_sigma (float): Standard deviation of the Gaussian blur kernel
            used to smooth the input images.
        k1 (float): One of the SSIM dampening parameters.
        k2 (float): One of the SSIM dampening parameters.

    Returns:
        jnp.ndarray: SSIM in range [0, 1] of shape ().
    """

    img0 = torch.as_tensor(img0).detach().cpu()
    img1 = torch.as_tensor(img1).detach().cpu()


    if mask is None:
        # mask = jnp.ones_like(img0[..., :1])
        mask = torch.ones_like(img0[..., :1])
    mask = mask[..., 0]  # type: ignore

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    # f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    f_i = ((torch.arange(filter_size).cpu() - hw + shift) / filter_sigma) ** 2
    # filt = jnp.exp(-0.5 * f_i)
    filt = torch.exp(-0.5 * f_i)
    # filt /= jnp.sum(filt)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # NOTICE Dusan: previous version used vectorization on Color channel, we need to avoid this
    def convolve2d(z, m, f):
        z_ = []
        for i in range(3):
            z_.append(torch.as_tensor(signal.convolve2d(z[...,i] * m, f, mode="valid")).cpu())
        z_ = torch.stack(z_, axis=-1)

        m_ = torch.as_tensor(signal.convolve2d(m, torch.ones_like(f), mode="valid")).cpu()

        return_where = []
        for i in range(3):
            return_where.append(torch.where(m_ != 0, z_[...,i] * torch.ones_like(f).sum() / m_, torch.tensor(0., device='cpu')))

        return_where = torch.stack(return_where, axis=-1)

        return return_where, (m_ != 0).type(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
    filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(img0, mask)[0]
    mu1 = filt_fn(img1, mask)[0]
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, mask)[0] - mu00
    sigma11 = filt_fn(img1**2, mask)[0] - mu11
    sigma01 = filt_fn(img0 * img1, mask)[0] - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    # sigma00 = jnp.maximum(0.0, sigma00)
    sigma00 = torch.maximum(torch.tensor(0.0).cpu(), sigma00)
    # sigma11 = jnp.maximum(0.0, sigma11)
    sigma11 = torch.maximum(torch.tensor(0.0).cpu(), sigma11)
    # sigma01 = jnp.sign(sigma01) * jnp.minimum(
        # jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
    # )
    sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = ssim_map.mean()

    return ssim


def eval_imgs(pred, gt, loss_fn_vgg, scale_ssim=100., scale_lpips=100.):
    pred = torch.from_numpy(pred).float()/255. # H,W,3
    gt = torch.from_numpy(gt).float()/255. # H,W,3
    pred = pred.cuda()
    gt = gt.cuda()

    metric_psnr = compute_psnr(pred, gt).cpu()
    metric_ssim = compute_ssim(pred, gt).cpu() * scale_ssim
    metric_lpips = eval_lpips(pred, gt, loss_fn_vgg) * scale_lpips
    return dict(psnr=metric_psnr, ssim=metric_ssim, lpips=metric_lpips)

def eval_lpips(img0, img1, loss_fn_vgg):
    # normalize images from [0,1] range to [-1,1]
    img0 = img0 * 2.0 - 1.0
    img1 = img1 * 2.0 - 1.0
    img0 = img0.unsqueeze(0).permute(0, 3, 1, 2)
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    return loss_fn_vgg(img0, img1).cpu()

@torch.no_grad()
def eval_all(src_dir, scale_ssim=100., scale_lpips=100.):
    results = collections.defaultdict(list)
    gt_dir = os.path.join(src_dir, 'gt')
    pred_dir = os.path.join(src_dir, 'renders')
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda().eval()

    gt_img_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')) + glob.glob(os.path.join(gt_dir, '*.jpg')))
    pred_img_paths = sorted(glob.glob(os.path.join(pred_dir, '*.png')) + glob.glob(os.path.join(pred_dir, '*.jpg')))
    assert len(gt_img_paths) == len(pred_img_paths), f'Number of images in gt and pred directories do not match: {len(gt_img_paths)} vs {len(pred_img_paths)}'

    for gt_img_path, img_path in tqdm(zip(gt_img_paths, pred_img_paths), total=len(gt_img_paths)):
        assert os.path.basename(gt_img_path) == os.path.basename(img_path), f'Image names do not match: {gt_img_path} vs {img_path}'
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_img_path)
        _eval = eval_imgs(img, gt, loss_fn_vgg, scale_ssim=scale_ssim, scale_lpips=scale_lpips)
        for key, val in _eval.items():
            results[key].append(val)
    for key, val in results.items():
        print(key, '=', torch.stack(val).mean().item())

    dst_results = os.path.join(src_dir, 'results.yaml')
    with open(dst_results, 'w') as f:
        f.write(f'ssim: {torch.stack(results["ssim"]).mean().item()}\n')
        f.write(f'psnr: {torch.stack(results["psnr"]).mean().item()}\n')
        f.write(f'lpips: {torch.stack(results["lpips"]).mean().item()}\n')
    print('Saved results to', dst_results)

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    import matplotlib
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().cpu().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    depth = 1-depth
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def viz_depth(depth, mask=None, min_depth=9, max_depth=100):
    depth = depth.squeeze() # H,W
    depth_vis = torch.clip(depth, min_depth, max_depth)
    # depth_vis = depth_vis.unsqueeze(0).repeat_interleave(3, dim=0).clone()
    depth_vis = (depth_vis-min_depth) / (max_depth-min_depth)
    depth_vis = torch.clip(depth_vis, 0.0, 1.0)
    _depth_vis = depth_vis.cpu().numpy()[...,None].repeat(3, axis=-1) # H,W,3
    _depth_vis = cv2.applyColorMap((_depth_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
    # mask = depth > 0
    if mask is not None:
        mask = mask.squeeze().clone().cpu().numpy()
        _depth_vis[~mask] = np.array([255,255,255], dtype=np.uint8)
    return _depth_vis
    depth_vis = torch.from_numpy(_depth_vis.transpose(2,0,1))
    return depth_vis

@torch.no_grad()
def render_set(model_path, load2gpu_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, is_static=False):
    results_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    rnd_depth = args.rnd_depth

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings, gts, gt_depths, depths = [], [], [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        # animate step
        gaussian_dict, _ = get_gaussian_dict(view, gaussians, deform, None, None, static=is_static)

        # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussian_dict, pipeline, background)
        rendering = results["render"]
        if rnd_depth:
            depth = results["depth"]

            gt_mask = view.mask
            depth_min, depth_max = 9, depth.max().item() + 1e-5
            if gt_mask is not None:
                gt_mask = gt_mask > 0
                if view.depth is not None:
                    gt_depth = view.depth[view.depth > 0]
                    # depth_min, depth_max = gt_depth.min().item(), gt_depth.max().item()
                    depth_max = gt_depth.max().item()
            depth_max = 10.5
            # depth_img = colorize_depth_maps(depth.cpu(), depth_min, depth_max, valid_mask=gt_mask)
            gt_depth_img = viz_depth(view.depth, mask=gt_mask, min_depth=depth_min, max_depth=depth_max)[:,:,::-1]
            rnd_depth_img = viz_depth(depth, mask=gt_mask, min_depth=depth_min, max_depth=depth_max)[:,:,::-1]
            # depth_img = depth / (depth.max() + 1e-5)
            cv2.imwrite(os.path.join(gt_depth_path, '{0:05d}'.format(idx) + ".png"), gt_depth_img)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), rnd_depth_img)

            depths.append(rnd_depth_img)
            gt_depths.append(gt_depth_img)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if view.original_image is not None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gts.append(to8b(gt.cpu().numpy()))
        # torchvision.utils.save_image(depth_img/255., os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    if len(gts) > 0:
        gts = np.stack(gts, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(gts_path, 'video.mp4'), gts, fps=30, quality=8)
    if rnd_depth:
        imageio.mimwrite(os.path.join(depth_path, 'video.mp4'), depths, fps=30, quality=8)
        imageio.mimwrite(os.path.join(gt_depth_path, 'video.mp4'), gt_depths, fps=30, quality=8)

    print('Saved', os.path.join(render_path, 'video.mp4'))
    # evaluate test images
    if name == 'test':
        eval_all(results_path)
    elif name == 'train':
        eval_all(results_path)



def render_sets(dataset: ModelParams, hyper: ModelHiddenParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, skip_pred: bool):
    setattr(hyper, 'n_frames', dataset.load_time_step if dataset.load_time_step > 1 else 0)
    is_static = dataset.is_static
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.use_isotropic = hyper.use_isotropic
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = SplatFieldsModel(hyper, radius=scene.cameras_extent)
        deform.load_weights(dataset.model_path, iteration=iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        if not skip_train:
            render_set(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, deform, is_static=is_static)

        if not skip_test:
            render_set(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, deform, is_static=is_static)

        if not skip_pred:
            render_set(dataset.model_path, dataset.load2gpu_on_the_fly, "pred", scene.loaded_iter,
                       scene.getPredCameras(), gaussians, pipeline,
                       background, deform, is_static=is_static)


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(0,120)] + [100_000, 200_000])
    parser.add_argument("--configs", type=str, default = "")
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), hp.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_pred)
