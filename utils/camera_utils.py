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

from scene.cameras import Camera, CameraPenoptic
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json
from PIL import Image
WARNED = False


def loadCam(args, id, cam_info, resolution_scale, max_resolution=800):
    if cam_info.image is None: # for visualization cameras
        _cam = Camera(
            colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            image=cam_info.image, gt_alpha_mask=cam_info.mask,
            image_name=cam_info.image_name, uid=id, fid=cam_info.fid,
            data_device=args.data_device, mask=cam_info.mask, 
            image_width=cam_info.width, image_height=cam_info.height)
        return _cam
    orig_w, orig_h = cam_info.image.size
    mask = cam_info.mask

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > max_resolution:
                global WARNED
                if not WARNED:
                    print(f"[ INFO ] Encountered quite large input images (>{max_resolution} pixels width), rescaling to {max_resolution}.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / max_resolution
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if mask is not None:
        # numpy to pil 
        mask_pil = Image.fromarray(np.uint8(np.squeeze(mask)*255))
        mask = PILtoTorch(mask_pil, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    is_penoptic = getattr(cam_info, 'penoptic', False)
    if is_penoptic:
        cls = CameraPenoptic
    else:
        cls = Camera
    return cls(
        colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, w2c=cam_info.w2c, w=cam_info.width, h=cam_info.height,
        FoVx=cam_info.FovX, FoVy=cam_info.FovY,
        image=gt_image, gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name, uid=id,
        data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
        depth=cam_info.depth, 
        mask=mask, 
        rays_o=cam_info.rays_o, 
        rays_d=cam_info.rays_d,
        fx=cam_info.fx, fy=cam_info.fy, cx=cam_info.cx, cy=cam_info.cy,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
