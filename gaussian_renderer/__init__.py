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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, gaussian_dict: dict, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, return_opacity=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means3D = gaussian_dict['means3D']
    active_sh_degree = gaussian_dict['active_sh_degree']
    gaussian_opacity = gaussian_dict['gaussian_opacity']
    gaussian_scales = gaussian_dict['gaussian_scales']
    gaussian_rotations = gaussian_dict['gaussian_rotations']
    gaussian_features = gaussian_dict.get('gaussian_features', None)
    gaussian_rgb = gaussian_dict.get('gaussian_rgb', None)
    if gaussian_rgb is None and 'gaussian_rgb_fnc' in gaussian_dict:
        ray_d = means3D - viewpoint_camera.camera_center[None]
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        gaussian_rgb = gaussian_dict['gaussian_rgb_fnc'](ray_d)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    raster_settings_mask = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer_mask = GaussianRasterizer(raster_settings=raster_settings_mask)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=gaussian_features,
        colors_precomp=gaussian_rgb,
        opacities=gaussian_opacity,
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        cov3D_precomp=None)
    opacity_image = None
    if return_opacity:
        # means2D_opacity = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        opacity_image = rasterizer_mask(
            means3D=means3D,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=torch.ones(gaussian_opacity.shape[0], 3, device=gaussian_opacity.device), # seems the right way...
            opacities=gaussian_opacity,
            scales=gaussian_scales,
            rotations=gaussian_rotations,
            cov3D_precomp=None)[0]
        opacity_image = opacity_image[:1] # 1,H,W

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            'opacity': opacity_image,
            "depth": depth}
