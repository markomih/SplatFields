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

import os
import sys
import trimesh
import torch
import tempfile
import uuid
from glob import glob
import cv2 as cv
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
import torch.nn.functional as F
from utils.camera_utils_multinerf import generate_interpolated_path
from tqdm import tqdm
from sklearn.cluster import KMeans

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample, random_state=0).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    camera_id: Optional[int] = None
    depth: Optional[np.array] = None
    mask: Optional[np.array] = None
    KRT: Optional[np.array] = None
    K: Optional[np.array] = None
    w2c: Optional[np.array] = None
    pose: Optional[np.array] = None
    rays_o: Optional[np.array] = None
    rays_d: Optional[np.array] = None
    fx: Optional[np.array] = None
    fy: Optional[np.array] = None
    cx: Optional[np.array] = None
    cy: Optional[np.array] = None
    penoptic: Optional[bool] = False


def get_ray_directions(H, W, K, OPENGL_CAMERA=False):
    x, y = torch.meshgrid(
        torch.arange(W, device=K.device),
        torch.arange(H, device=K.device),
        indexing="xy",
    )
    camera_dirs = F.pad(torch.stack([
            (x - K[0, 2] + 0.5) / K[0, 0],
            (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
        ], dim=-1,), (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]

    return camera_dirs

def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    pred_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder=None, white_background=False):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(sorted(cam_extrinsics)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        if masks_folder is not None:
            mask_name = extr.name[1:]
            mask_path = os.path.join(masks_folder, mask_name)
            mask = Image.open(mask_path)
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        else:
            mask = None

        try:
            fid = int(image_name) / (num_frames - 1)
        except:
            fid = 0
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid, mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def read_colmap_poses(path, images, white_background):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), masks_folder=os.path.join(path, 'mask'), white_background=white_background)
    camera_pose = []
    for cam in cam_infos_unsorted:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = cam.R.transpose()
        Rt[:3, 3] = cam.T
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        camera_pose.append(cam_center.reshape(-1))
    # camera_pose = [cam.T for cam in cam_infos_unsorted]
    cam_pos = np.stack(camera_pose)
    return cam_infos_unsorted, cam_pos

def readColmapSceneInfoSparse(path, images, eval, white_background, llffhold=8, num_pts=300_000, pc_path='', load_time_step=10000, load_every_nth=-1, n_views=6):
    cam_infos_unsorted, _cam_pose = read_colmap_poses(path, images, white_background)
    # pixel nerf split
    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
    split_indices = {'test': test_idx, 'train': train_idx}

    # selected_idxs = sorted(kmeans_downsample(_cam_pose, n_views))
    selected_idxs = split_indices['train'][:n_views]
    print('training camera ids', selected_idxs, ':', [cam_infos_unsorted[c].image_name for c in selected_idxs])

    train_cam_infos, test_cam_infos = [], []
    for ind in range(len(cam_infos_unsorted)):
        if ind in selected_idxs:
            train_cam_infos.append(cam_infos_unsorted[ind])
        elif ind in test_idx:
            test_cam_infos.append(cam_infos_unsorted[ind])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if pc_path is not None and pc_path != '':
        assert os.path.exists(pc_path), f"Path {pc_path} does not exist"
        xyz = np.asarray(trimesh.load(pc_path).vertices)
        # remove points that are outside -1,1 range
        xyz = xyz[np.all(np.abs(xyz) < 1, axis=1)]
        # subsample vertices to 100000 points
        if num_pts > 0 and xyz.shape[0] > num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
    else:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, colors, _ = read_points3D_binary(bin_path)
        except:
            xyz, colors, _ = read_points3D_text(txt_path)
        # storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    storePly(ply_path, xyz, colors)
    pcd = fetchPly(ply_path)
    print('Using', len(train_cam_infos), 'training cameras. Using', len(test_cam_infos), 'validation cameras')
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(
        path,
        images,
        eval,
        white_background,
        llffhold=8,
        num_pts=100_000,
        load_time_step=10000,
        load_every_nth=1,
        pc_path=None,
        n_views=-1,
        ):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir), masks_folder=os.path.join(path, 'mask'), white_background=white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate( cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate( cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    train_cam_infos = train_cam_infos[:load_time_step]
    train_cam_infos = train_cam_infos[::load_every_nth]
    print('Loading', len(train_cam_infos), 'Training Cameras')
    ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")
    if pc_path is not None and pc_path != '':
        assert os.path.exists(pc_path), f"Path {pc_path} does not exist"
        xyz = np.asarray(trimesh.load(pc_path).vertices)
        # subsample vertices to 100000 points
        if num_pts > 0 and xyz.shape[0] > num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
    else:
        # ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, colors, _ = read_points3D_binary(bin_path)
        except:
            xyz, colors, _ = read_points3D_text(txt_path)
        # storePly(ply_path, xyz, rgb)
        # pcd = fetchPly(ply_path)
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    storePly(ply_path, xyz, colors)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path
                           )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", load_time_step=10000):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"][:load_time_step]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame.get('time', 0)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time, mask=mask))

    return cam_infos

def readCamerasFromTransformsCV(path, transformsfile, white_background, extension=".png", load_time_step=10000):
    cam_infos = []
    BLENDER_TO_OPENCV_MATRIX = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)
    model_scale_dict = dict(chair=2.1, drums=2.3, ficus=2.3, hotdog=3.0, lego=2.4, materials=2.4, mic=2.5, ship=2.75)
    obj_name = os.path.basename(path)
    world_scale = 2 / model_scale_dict[obj_name]
    cam_pos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"][:load_time_step]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame.get('time', 0)
            transform_matrix = np.array(frame["transform_matrix"])
            transform_matrix = (transform_matrix @ BLENDER_TO_OPENCV_MATRIX)
            transform_matrix[:3, :4] *= world_scale
            cam_pos.append(transform_matrix[:3, 3])

            w2c = np.linalg.inv(np.array(transform_matrix))
            R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            # FovY = fovx
            # FovX = fovy
            # focal_length = image.size[1] / np.tan(contents["camera_angle_x"] / 2)
            # FovY = focal_length
            # FovX = focal_length
            w, h = image.size
            x, y = w / 2, h / 2
            focal_length = y / np.tan(contents['camera_angle_x'] / 2)
            FovY = focal2fov(focal_length, h)
            FovX = focal2fov(focal_length, w)
            K = np.array([[focal_length, 0, x], [0, focal_length, y], [0, 0, 1]])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, K=K, pose=transform_matrix[:3, :4],
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time, mask=mask))
    cam_pos = np.stack(cam_pos, axis=0)
    return cam_infos, cam_pos

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def readNerfSyntheticInfo(
        path,
        white_background,
        eval,
        extension=".png",
        load_time_step=10000,
        num_pts=100_000,
        max_num_pts=-1,
        pts_samples='load',
        pc_path='/media/STORAGE_4TB/projects/forkforkDeformable-3D-Gaussians/output_rep/nerf_synthetic/lego/static10/point_cloud/iteration_7000/point_cloud.ply'
        ):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension, load_time_step=load_time_step)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if pts_samples == 'load':
        import trimesh
        assert os.path.exists(pc_path), f"Path {pc_path} does not exist"
        xyz = np.asarray(trimesh.load(pc_path).vertices)
        # remove points outside the visual hull 
        xyz_mask = np.ones_like(xyz[:, 0], dtype=bool)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in train_cam_infos:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T

            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
            if False:
                uv = np.round(uv.numpy()).astype(int)
                image = np.array(cam.image)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                # set pixels to 0 if they are not in the mask
                image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                # save image
                imageio.imsave(f'./uv_img.png', image)
                print('saved image', f'./uv_img.png')
                import pdb; pdb.set_trace()
            uv = np.round(uv.numpy()).astype(int)
            _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < cam.image.size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.image.size[0])
            uv = uv[_pix_mask]
            cam_mask = np.array(cam.mask)
            _pix_mask[_pix_mask] = cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0

                
            xyz_mask = xyz_mask & _pix_mask
        if False:
            # tmp save point cloud for debugging
            import trimesh 
            trimesh.PointCloud(xyz[xyz_mask]).export('tmp.ply')
            import pdb; pdb.set_trace()
        xyz = xyz[xyz_mask]
        if max_num_pts > 0 and xyz.shape[0] > max_num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], max_num_pts, replace=False)]
        # subsample vertices to 100000 points
        # xyz = xyz[np.random.choice(xyz.shape[0], 200_000, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
    elif pts_samples == 'random':

        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        colors = np.random.random((num_pts, 3)) / 255.0
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
        #     shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif pts_samples == 'hull':
        aabb = -1.5, 1.5
        grid_resolution = 256
        grid = np.linspace(aabb[0], aabb[1], grid_resolution)
        grid = np.meshgrid(grid, grid, grid)
        grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

        # project grid locations to the image plane
        grid = torch.from_numpy(np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1)).float() # n_pts, 4
        grid_mask = np.ones_like(grid_loc[:, 0], dtype=bool)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in train_cam_infos:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = grid @ full_proj_transform # (full_proj_transform @ xyzh.T).T
            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))

            uv = np.round(uv.numpy()).astype(int)
            _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < cam.image.size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.image.size[0])
            uv = uv[_pix_mask]
            cam_mask = np.array(cam.mask)
            _pix_mask[_pix_mask] = cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0
            grid_mask = grid_mask & _pix_mask

        xyz = grid[:, :3].numpy()[grid_mask]
        if False:
            # tmp save point cloud for debugging
            import trimesh 
            trimesh.PointCloud(xyz).export('tmp_xyzhull.ply')
            import pdb; pdb.set_trace()

        if xyz.shape[0] > num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
        # 
    else:
        raise NotImplementedError
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, colors)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfSyntheticCVInfo(
        path,
        white_background,
        eval,
        extension=".png",
        load_time_step=10000,
        n_views=6,
        num_pts=100_000,
        max_num_pts=-1,
        pts_samples='load',
        depth_path='/media/STORAGE_4TB/projects/zerorf/results/test/hotdog/depth/000006_views.pt',
        pc_path='/media/STORAGE_4TB/projects/forkforkDeformable-3D-Gaussians/output_rep/nerf_synthetic/lego/static10/point_cloud/iteration_7000/point_cloud.ply'
        ):
    print("Reading Training Transforms")
    train_cam_infos, _cam_pose = readCamerasFromTransformsCV(
        # path, "transforms_train.json", white_background, extension, load_time_step=load_time_step)
        path, "transforms_train.json", white_background, extension)
    selected_idxs = sorted(kmeans_downsample(_cam_pose, n_views))
    train_cam_infos = [train_cam_infos[i] for i in selected_idxs]

    print("Reading Test Transforms")
    test_cam_infos, _ = readCamerasFromTransformsCV(
        path, "transforms_test.json", white_background, extension)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if pts_samples == 'depth':
        # load depths 
        depth_maps_dict = torch.load(depth_path)
        depths = torch.stack([depth_maps_dict[_i] for _i in selected_idxs])
        # unproject depths to 3D points
        if True:
            _pts, _rgb = [], []
            import pdb; pdb.set_trace()
            for ind, cam in enumerate(train_cam_infos):
                K = cam.K # (3,3)
                H, W = cam.image.size
                directions = get_ray_directions(H, W, torch.from_numpy(cam.K).float()) # (h, w, 3)
                rays_o, rays_d = get_rays(directions, torch.from_numpy(cam.pose).float())

                _pts.append((rays_o + rays_d * depths[ind].view(-1, 1)).numpy())
                if False:
                    c2w = cam.pose # (3,4)
                    cam_depth = depths[ind].squeeze(0) # (H,W)
                    H, W = cam.image.size
                    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy",)
                    x, y = x.reshape(-1), y.reshape(-1)
                    _d = cam_depth[y, x]
                    xyd = torch.stack([x, y, _d], dim=-1).numpy()  # (3,H*W)
                    _dmask = (_d > 0) & (_d < 100)
                    xyd = xyd[_dmask]  # (3,n)
                    _rgb.append(np.asarray(cam.image)[y[_dmask], x[_dmask]])

                    pts = np.linalg.inv(K) @ xyd.T # (3,n)
                    pts = c2w @ np.concatenate([pts, np.ones_like(pts[:1])])
                    pts = pts.T
                    _pts.append(pts)

            pts = np.concatenate(_pts, axis=0)
            # rgb = (np.concatenate(_rgb, axis=0)*255).astype(np.uint8)
            # rgb = np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=-1)
            import trimesh
            import pdb; pdb.set_trace()
            _ = trimesh.PointCloud(vertices=pts).export(f'tmp_hotdog_depth.ply')
            # _ = trimesh.PointCloud(vertices=pts,  colors=rgb).export(f'tmp_hotdog_depth.ply')
            print('Saved:', f'tmp_hotdog_depth.ply')
            import pdb; pdb.set_trace()
            exit(0)
    elif pts_samples == 'load':
        import trimesh
        assert os.path.exists(pc_path), f"Path {pc_path} does not exist"
        xyz = np.asarray(trimesh.load(pc_path).vertices)
        # remove points outside the visual hull 
        xyz_mask = np.ones_like(xyz[:, 0], dtype=bool)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in train_cam_infos:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T

            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
            if False:
                uv = np.round(uv.numpy()).astype(int)
                image = np.array(cam.image)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                # set pixels to 0 if they are not in the mask
                image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                # save image
                imageio.imsave(f'./uv_img.png', image)
                print('saved image', f'./uv_img.png')
                import pdb; pdb.set_trace()
            uv = np.round(uv.numpy()).astype(int)
            _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < cam.image.size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.image.size[0])
            uv = uv[_pix_mask]
            cam_mask = np.array(cam.mask)
            _pix_mask[_pix_mask] = cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0

                
            xyz_mask = xyz_mask & _pix_mask
        if False:
            # tmp save point cloud for debugging
            import trimesh 
            trimesh.PointCloud(xyz[xyz_mask]).export('tmp_hotdog_load.ply')
            import pdb; pdb.set_trace()
        xyz = xyz[xyz_mask]
        if max_num_pts > 0 and xyz.shape[0] > max_num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], max_num_pts, replace=False)]
        # subsample vertices to 100000 points
        # xyz = xyz[np.random.choice(xyz.shape[0], 200_000, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
    elif pts_samples == 'random':

        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        colors = np.random.random((num_pts, 3)) / 255.0
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
        #     shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif pts_samples == 'hull':
        aabb = -1.0, 1.0
        grid_resolution = 256
        grid = np.linspace(aabb[0], aabb[1], grid_resolution)
        grid = np.meshgrid(grid, grid, grid)
        grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

        # project grid locations to the image plane
        grid = torch.from_numpy(np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1)).float() # n_pts, 4
        grid_mask = np.ones_like(grid_loc[:, 0], dtype=bool)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in train_cam_infos:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = grid @ full_proj_transform # (full_proj_transform @ xyzh.T).T
            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))

            uv = np.round(uv.numpy()).astype(int)
            _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < cam.image.size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.image.size[0])
            uv = uv[_pix_mask]
            cam_mask = np.array(cam.mask)
            _pix_mask[_pix_mask] = cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0
            grid_mask = grid_mask & _pix_mask

        xyz = grid[:, :3].numpy()[grid_mask]
        if False:
            for cam in train_cam_infos:
                import pdb; pdb.set_trace()
                world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
                projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
                cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T
                uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
                uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
                if True:
                    uv = np.round(uv.numpy()).astype(int)
                    image = np.array(cam.image)
                    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                    # set pixels to 0 if they are not in the mask
                    image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                    # save image
                    imageio.imsave(f'./uv_img.png', image)
                    print('saved image', f'./uv_img.png')
                    import pdb; pdb.set_trace()

        if False:
            # tmp save point cloud for debugging
            import trimesh 
            trimesh.PointCloud(xyz).export('tmp_hotdog_hull.ply')
            import pdb; pdb.set_trace()

        if xyz.shape[0] > num_pts:
            xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
        # 
    else:
        raise NotImplementedError
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, colors)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        try:
            fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        except:
            fid = 0
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid, mask=mask[:, :, :1])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if False:
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        xyz = pcd.points
        for cam in train_cam_infos:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T

            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
            if True:
                uv = np.round(uv.numpy()).astype(int)
                image = np.array(cam.image)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                # set pixels to 0 if they are not in the mask
                image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                # save image
                imageio.imsave(f'./uv_img.png', image)
                print('saved image', f'./uv_img.png')
                import pdb; pdb.set_trace()


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           pred_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        mask_path = image_path.replace('/rgb/', '/mask/') + '.png'
        mask = np.array(Image.open(mask_path)) / 255.0

        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask[..., None],
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval, pc_path=''):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if pc_path != '':
        assert os.path.exists(pc_path), f"Path {pc_path} does not exist"
        xyz = np.asarray(trimesh.load(pc_path).vertices)
        # sample 100000 points
        if xyz.shape[0] > 200_000:
            xyz = xyz[np.random.choice(xyz.shape[0], 200_000, replace=False)]

        # remove points outside the visual hull
        # xyz = (xyz - scene_center) * scene_scale
        if False:
            xyz = torch.from_numpy(xyz).float()
            zfar = 100.0
            znear = 0.01
            trans=np.array([0.0, 0.0, 0.0])
            scale=1.0
            for cam in train_cam_infos:
                world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale))#.transpose(0, 1)
                projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY)#.transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
                cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T
                uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
                W, H = cam.image.size
                uv = ndc2Pix(uv, np.array([W, H]))

                uv = np.round(uv.numpy()).astype(int)
                _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < cam.image.size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.image.size[0])
                uv = uv[_pix_mask]
                import pdb; pdb.set_trace()
                cam_mask = np.array(cam.mask)
                _pix_mask[_pix_mask] = cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0
                grid_mask = grid_mask & _pix_mask
                if True:
                    uv = np.round(uv.numpy()).astype(int)
                    image = np.array(cam.image)
                    import pdb; pdb.set_trace()
                    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                    # set pixels to 0 if they are not in the mask
                    bg_uv = uv[cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) == 0]
                    fg_uv = uv[cam_mask[uv[:, 1], uv[:, 0]].reshape(-1) > 0]
                    image[bg_uv[:, 1], bg_uv[:, 0]] = np.array([255, 0, 0])
                    image[fg_uv[:, 1], fg_uv[:, 0]] = np.array([0, 255, 0])

                    # save image
                    imageio.imsave(f'./uv_img.png', image)
                    print('saved image', f'./uv_img.png')
                    import pdb; pdb.set_trace()

            xyz = xyz.numpy()[grid_mask]

        colors = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
        ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")
        storePly(ply_path, xyz, colors)
        pcd = fetchPly(ply_path)

        ply_path = pc_path
    else:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            print(f"Generating point cloud from nerfies...")

            xyz = np.load(os.path.join(path, "points.npy"))
            xyz = (xyz - scene_center) * scene_scale
            num_pts = xyz.shape[0]
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def parse_cam(scale_mats_np, world_mats_np):
    intrinsics_all, pose_all = [], []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
    return np.stack(intrinsics_all), np.stack(pose_all) # [n_images, 4, 4]

def readCamerasFromNeus(data_dir, white_background, _sample = lambda x, y: x, fid=None, penoptic=False):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(data_dir)
    print('Load data:', data_dir)
    _images_lis = sorted(glob(os.path.join(data_dir, 'image/*.png')) + glob(os.path.join(data_dir, 'rgb/*.png')) + glob(os.path.join(data_dir, 'rgb/*.jpg')))
    _frame_ids = [int(os.path.splitext(os.path.basename(im_name))[0]) for im_name in _images_lis]
    _camera_dict = np.load(os.path.join(data_dir, 'cameras_sphere.npz'))
    world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in _frame_ids], _frame_ids) # world_mat is a projection matrix from world to image
    scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in _frame_ids], _frame_ids) # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    # world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))], _frame_ids) # world_mat is a projection matrix from world to image
    # scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))], _frame_ids) # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    intrinsics_all, pose_all = parse_cam(scale_mats_np, world_mats_np)

    images_lis = _sample(_images_lis, _frame_ids)
    masks_lis = _sample(sorted(glob(os.path.join(data_dir, 'mask/*.png'))), _frame_ids)

    images = np.stack([cv.imread(im_name)[..., ::-1] for im_name in images_lis]) / 255.0  # [n_images, H, W, 3]
    all_c2w = pose_all[:, :3, :4].astype(np.float32)
    all_w2c = np.linalg.inv(pose_all.astype(np.float32))[:, :3, :4]
    all_images = images.astype(np.float32)

    has_masks = len(masks_lis) > 0
    if has_masks:
        masks  = np.stack([cv.imread(im_name) for im_name in masks_lis]) / 255.0   # [n_images, H, W, 3]
        if len(masks.shape) == 4:
            masks = masks[..., 0:1]   # [n_images, H, W, 1]
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        all_images = all_images * masks + (1 - masks) * bg

    depth_lis = _sample(sorted(glob(os.path.join(data_dir, 'depth/*.png'))), _frame_ids)
    has_depth = len(depth_lis) > 0
    if has_depth:
        depth_scale = 1000. #config.get('depth_scale', 1000.)
        depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in depth_lis]) / depth_scale
        depths_np = depths_np*(1./scale_mats_np[0][0, 0])
        depths_np[depths_np == 0] = -1. # avoid nan values
        depths = depths_np.astype(np.float32)
        if has_masks:
            depths[~(masks[..., 0] > 0)] = -1
    # unproject depth to 3D points

    h, w = all_images.shape[1:-1]
    intrinsics_all_inv = torch.inverse(torch.from_numpy(intrinsics_all)).float()

    pc_list = []
    # create scene info
    KRT = intrinsics_all[:, :3, :3] @ all_w2c
    cam_infos = []
    num_frames = all_images.shape[0]
    H,W = all_images.shape[1:3]
    directions = get_ray_directions(H, W, torch.from_numpy(intrinsics_all[0]).float()) # (h, w, 3)
    for cam_id in range(all_images.shape[0]):
        w2c = all_w2c[cam_id]
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
        K = intrinsics_all[cam_id]
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, h)
        FovX = focal2fov(focal_length_x, w)

        img_np = (all_images[cam_id]*255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(img_np, "RGB")
        image_name = os.path.splitext(os.path.basename(images_lis[cam_id]))[0]
        if fid is None:
            _fid = int(image_name) / max((num_frames - 1), 1)
        else:
            _fid = fid

        rays_o, rays_d = get_rays(directions, torch.from_numpy(all_c2w[cam_id]))
        cam_infos.append(CameraInfo(
            uid=cam_id,
            w2c=w2c,
            R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            fx=K[0,0], fy=K[1,1], cx=K[0, 2], cy=K[1, 2],
            image=image,
            depth=depths_np[cam_id] if has_depth else None,
            mask=masks[cam_id] if has_masks else None,
            image_path=images_lis[cam_id],
            image_name=Path(images_lis[cam_id]).stem, # images_lis[cam_id]
            width=image.size[0], 
            height=image.size[1],
            fid=_fid,
            KRT=KRT[cam_id],
            rays_o=rays_o.view_as(directions), # H,W,3
            rays_d=rays_d.view_as(directions), # H,W,3
            penoptic=penoptic,
        ))
        if has_depth:
            pc_list.append(_gen_3dpoints(cam_id, h, w, intrinsics_all_inv, torch.from_numpy(pose_all).float(), torch.from_numpy(depths), torch.from_numpy(img_np)))

    all_pc = None
    if has_depth:
        all_pc_xyz = np.concatenate([val[0] for val in pc_list])
        all_pc_color = np.concatenate([val[1] for val in pc_list])
        all_pc = (all_pc_xyz, all_pc_color)
    return cam_infos, all_pc

def visual_hull_samples(masks, KRT, n_pts=100_000, grid_resolution=256, aabb=(-1., 1.)):
    """ 
    Args:
        masks: (n_images, H, W)
        KRT: (n_images, 3, 4)
        grid_resolution: int
        aabb: (2)
    """
    # create voxel grid coordinates
    grid = np.linspace(aabb[0], aabb[1], grid_resolution)
    grid = np.meshgrid(grid, grid, grid)
    grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

    # project grid locations to the image plane
    grid = np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1) # n_pts, 4
    grid = grid[None].repeat(masks.shape[0], axis=0) # n_imgs, n_pts, 4
    grid = grid @ KRT.transpose(0, 2, 1)  # (n_imgs, n_pts, 4) @ (n_imgs, 4, 3) -> (n_imgs, n_pts, 3)
    uv = grid[..., :2] / grid[..., 2:] # (n_imgs, n_pts, 2)
    _, H, W = masks.shape[:3]  # n_imgs,H,W
    uv[..., 0] = 2.0 * (uv[..., 0] / (W - 1.0)) - 1.0
    uv[..., 1] = 2.0 * (uv[..., 1] / (H - 1.0)) - 1.0

    uv = torch.from_numpy(uv)
    masks = torch.from_numpy(masks)[:, None].squeeze(-1)
    samples = F.grid_sample(masks, uv[:, None], align_corners=True, mode='nearest', padding_mode='zeros').squeeze()
    _ind = (samples > 0).all(0) # (n_imgs, n_pts) -> (n_pts)

    # sample points around the grid locations
    grid_samples = grid_loc[_ind] # (n_pts, 2)
    all_samples = grid_samples
    np.random.shuffle(all_samples)

    return all_samples[:n_pts]

def visual_hull_samples_list(masks_list, KRT, n_pts=100_000, grid_resolution=256, aabb=(-1., 1.)):
    """ Visual hull from multiple views (images of different resolutions)
    Args:
        masks_list (list): n_images images of (H, W)
        KRT: (n_images, 3, 4)
        grid_resolution: int
        aabb: (2)
    """
    # create voxel grid coordinates
    grid = np.linspace(aabb[0], aabb[1], grid_resolution)
    grid = np.meshgrid(grid, grid, grid)
    grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

    # project grid locations to the image plane
    grid = np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1) # n_pts, 4
    grid = grid[None].repeat(KRT.shape[0], axis=0) # n_imgs, n_pts, 4
    grid = grid @ KRT.transpose(0, 2, 1)  # (n_imgs, n_pts, 4) @ (n_imgs, 4, 3) -> (n_imgs, n_pts, 3)
    uv = grid[..., :2] / grid[..., 2:] # (n_imgs, n_pts, 2)
    for i in range(len(masks_list)):
        H, W = masks_list[i].shape[:2]
        uv[i, ..., 0] = 2.0 * (uv[i, ..., 0] / (W - 1.0)) - 1.0
        uv[i, ..., 1] = 2.0 * (uv[i, ..., 1] / (H - 1.0)) - 1.0

    uv = torch.from_numpy(uv)
    outside_pts = (uv[..., 0] < -1) | (uv[..., 0] > 1) | (uv[..., 1] < -1) | (uv[..., 1] > 1)
    samples = []
    for i in range(len(masks_list)):
        mask = torch.from_numpy(masks_list[i][None])[:, None].squeeze(-1)
        samples.append(F.grid_sample(mask, uv[i][None][:, None], align_corners=True, mode='nearest', padding_mode='zeros').squeeze())
    samples = torch.stack(samples, dim=0)
    samples = samples > 0
    samples = samples | outside_pts
    _ind = samples.all(0) # (n_imgs, n_pts) -> (n_pts)

    # sample points around the grid locations
    grid_samples = grid_loc[_ind] # (n_pts, 2)
    all_samples = grid_samples
    np.random.shuffle(all_samples)

    return all_samples[:n_pts]

def readCamerasFromResFields(data_root, white_background, cam_names=['cam_train_1', 'cam_train_2'], _sample = lambda x: x, fid=None, penoptic=False):
    cam_infos_list, pc_list = [], []
    for cam_name in cam_names: 
        data_dir = os.path.join(data_root, cam_name)
        cam_infos, pc = readCamerasFromNeus(data_dir, white_background, _sample, fid=fid, penoptic=penoptic)
        cam_infos_list.extend(cam_infos)
        if pc is not None:
            pc_list.append(pc)

    all_pc = None
    if len(pc_list) > 0:
        all_pc_xyz = np.concatenate([val[0] for val in pc_list])
        all_pc_color = np.concatenate([val[1] for val in pc_list])
        all_pc = (all_pc_xyz, all_pc_color)
    return cam_infos_list, all_pc

def _gen_3dpoints(img_idx, H, W, intrinsics_all_inv, pose_all, depths, image):
    tx = torch.linspace(0, W - 1, W)
    ty = torch.linspace(0, H - 1, H)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
    p = torch.matmul(intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
    
    rays_o, rays_v = rays_o.transpose(0, 1), rays_v.transpose(0, 1)
    rays_depth = depths[img_idx][(pixels_y.long().reshape(-1), pixels_x.long().reshape(-1))].reshape(W, H).transpose(0, 1)
    dmask = rays_depth > 0
    pts = rays_o + rays_depth.unsqueeze(-1)*rays_v
    rgb = image[(pixels_y.long().reshape(-1), pixels_x.long().reshape(-1))].reshape(W, H, 3).transpose(0, 1)
    return pts[dmask].numpy(), rgb[dmask].numpy()

def readNeuSceneInfo(
        path,
        white_background,
        train_cam_names,
        test_cam_names,
        pred_cam_names,
        resfield=False,
        load_time_step=10000,
        num_pts=100_000,
        pts_samples='random'
    ):
    print("Reading Cameras")
    fid = 0 if load_time_step == 1 else None
    penoptic = False
    if pts_samples == 'vertices':
        penoptic = True

    if resfield:
        def _sample(data_list, fid_list):
            _toret = [_data for _data, _fid in zip(data_list, fid_list) if _fid < load_time_step]
            return _toret
        train_cam_infos, all_pc = readCamerasFromResFields(path, white_background, cam_names=train_cam_names, _sample=_sample, fid=fid, penoptic=penoptic)
        test_cam_infos, _ = readCamerasFromResFields(path, white_background, cam_names=test_cam_names, _sample=_sample, fid=fid, penoptic=penoptic)
        def _sample_pred(data_list, fid_list):
            _toret = [_data for _data, _fid in zip(data_list, fid_list)]
            return _toret
        pred_cam_infos, _ = readCamerasFromResFields(path, white_background, cam_names=pred_cam_names, _sample=_sample_pred, fid=fid, penoptic=penoptic)
    else:
        train_cam_infos, all_pc = readCamerasFromNeus(path, white_background, fid=fid, penoptic=penoptic)
        test_cam_infos = []
        pred_cam_infos = []
    # nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization = {'translate': np.array([ 0, 0, 0], dtype=np.float32), 'radius': 1.0}

    # if all_pc is not None:
    if pts_samples == 'vertices':
        vertices_path = os.path.join(path, 'vertices.npz')
        assert os.path.exists(vertices_path), f"Vertices file not found: {vertices_path}"
        _data = np.load(vertices_path)
        seg_mask = _data['seg'] == 1.0
        xyz = _data['vertices'][seg_mask]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0

        if False:
            for cam in train_cam_infos:
                KRT = cam.KRT
                uv = KRT @ np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1).T
                uv = uv[:2] / uv[2:3]
                uv = uv.T 
                uv = uv.astype(np.int32)
                img = np.array(cam.image)
                h,w= img.shape[:2]
                uv = np.clip(uv, 0, [w-1, h-1])
                img[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                img = Image.fromarray(img)
                img.save('test_uv.png')
                print('Saved test.png', 'test_uv.png')

        if False:
            zfar = 100.0
            znear = 0.01
            trans=np.array([0.0, 0.0, 0.0])
            scale=1.0

            if True:
                for cam in train_cam_infos:
                    world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
                    projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
                    cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T
                    uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
                    uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
                    if True:
                        uv = np.round(uv.numpy()).astype(int)
                        image = np.array(cam.image)
                        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                        # set pixels to 0 if they are not in the mask
                        image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                        # save image
                        imageio.imsave(f'./uv_img.png', image)
                        print('saved image', f'./uv_img.png')
                        import pdb; pdb.set_trace()

            # for cam in train_cam_infos:
            #     world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
            #     # my_w2c = cam.w2c
            #     import pdb; pdb.set_trace()
            #     projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
            #     full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            #     xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            #     cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T

            #     uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            #     uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
            #     if True:
            #         uv = np.round(uv.numpy()).astype(int)
            #         image = np.array(cam.image)
            #         uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
            #         # set pixels to 0 if they are not in the mask
            #         image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
            #         # save image
            #         imageio.imsave(f'./uv_img.png', image)
            #         print('saved image', f'./uv_img.png')
            #         import pdb; pdb.set_trace()

    elif pts_samples == 'random':
        # generate random points in range [-0.7, 0.7]
        xyz = np.random.random((num_pts, 3)) * 0.9 * 2.0 - 1.0
        colors = np.random.random((num_pts, 3)) / 255.0
    elif pts_samples == 'hull':
        if all_pc is None:
            aabb = -1.0, 1.0
        else:
            aabb = all_pc[0].min(), all_pc[0].max()
        # visual hull created only from the first frame
        KRT = np.stack([cam.KRT for cam in train_cam_infos if cam.fid == 0])
        train_cam_infos[1].mask.shape, train_cam_infos[2].mask.shape, train_cam_infos[3].mask.shape, train_cam_infos[4].mask.shape, train_cam_infos[5].mask.shape, train_cam_infos[6].mask.shape, train_cam_infos[7].mask.shape,   
        masks = [cam.mask for cam in train_cam_infos if cam.fid == 0]
        try:
            xyz = visual_hull_samples(np.stack(masks), KRT, n_pts=num_pts, grid_resolution=256, aabb=aabb)
        except Exception as e:
            if False:
                _tmp_m, _tmp_krt = [], []
                for _m, _k in zip(masks, KRT):
                    if _m.shape[0] != 512:
                        _tmp_m.append(_m)
                        _tmp_krt.append(_k)
                _tmp_m = np.stack(_tmp_m)
                _tmp_krt = np.stack(_tmp_krt)
                xyz = visual_hull_samples(_tmp_m, _tmp_krt, n_pts=num_pts, grid_resolution=256, aabb=aabb)
            else:
                xyz = visual_hull_samples_list(masks, KRT, n_pts=num_pts, grid_resolution=256, aabb=aabb)
        if False:
            zfar = 100.0
            znear = 0.01
            trans=np.array([0.0, 0.0, 0.0])
            scale=1.0

            if True:
                for cam in train_cam_infos:
                    world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)
                    projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY).transpose(0, 1)
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
                    cam_xyz = xyzh @ full_proj_transform # (full_proj_transform @ xyzh.T).T
                    uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
                    uv = ndc2Pix(uv, np.array([cam.image.size[1], cam.image.size[0]]))
                    if True:
                        uv = np.round(uv.numpy()).astype(int)
                        image = np.array(cam.image)
                        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
                        # set pixels to 0 if they are not in the mask
                        image[uv[:, 1], uv[:, 0]] = np.array([255, 0, 0])
                        # save image
                        imageio.imsave(f'./uv_img.png', image)
                        print('saved image', f'./uv_img.png')
                        import pdb; pdb.set_trace()

        colors = np.random.random((xyz.shape[0], 3)) / 255.0
    elif pts_samples == 'depth':
        assert all_pc is not None
        xyz, colors = all_pc[0], all_pc[1]
        if xyz.shape[0] > num_pts:
            _ind = np.random.choice(xyz.shape[0], num_pts, replace=False)
            xyz, colors = xyz[_ind], colors[_ind]
    else:
        raise NotImplementedError
    # shs = np.random.random((xyz.shape[0], 3)) / 255.0
    # colors=SH2RGB(shs)
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, colors)
    pcd = fetchPly(ply_path)
    # else:
    #     # if not os.path.exists(ply_path):
    #     if True:
    #         # Since this data set has no colmap data, we start with random points
    #         print(f"Generating random point cloud ({num_pts})...")
            
    #         # We create random points inside the bounds of the synthetic Blender scenes
    #         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #         shs = np.random.random((num_pts, 3)) / 255.0
    #         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #         storePly(ply_path, xyz, SH2RGB(shs) * 255)
    #     try:
    #         pcd = fetchPly(ply_path)
    #     except:
    #         pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=pred_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
        
def readResFieldSceneInfo(path, white_background, train_cam_names, test_cam_names,  pred_cam_names, load_time_step=10000, num_pts=100_000, pts_samples='random'):
    return readNeuSceneInfo(path, white_background, train_cam_names, test_cam_names,  pred_cam_names, resfield=True, load_time_step=load_time_step, num_pts=num_pts, pts_samples=pts_samples)

def readNerfiesCameras_mv(path, load_time_step=10000):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        # ratio = 0.25
        ratio = 1.0
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    # all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    camera_ids = [meta_json[i]['camera_id'] for i in all_img]
    # filter out based on time
    if load_time_step < np.max(all_time):
        selected_samples = [_ind for _ind, _t in enumerate(all_time) if _t < load_time_step]
        selected_train_samples = [_ind for _ind, _t in enumerate(all_time[:train_num]) if _t < load_time_step]
        train_num = len(selected_train_samples)
        all_img = [all_img[i] for i in selected_samples]
        all_time = [all_time[i] for i in selected_samples]
        camera_ids = [camera_ids[i] for i in selected_samples]
    
    # all_time = [_t for _t in all_time if _t < load_time_step]
    max_time = max(max(all_time), 1)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    camera_dict = {}
    for idx in tqdm(range(len(all_img))):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, #PILtoTorch(image, None),
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid, camera_id=camera_ids[idx]
                              )
        if fid == 0:
            camera_dict[camera_ids[idx]] = cam_info
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale, camera_dict

def readNerfiesInfo_mv(path, eval, load_time_step=10000, max_pts=300000):
    import trimesh
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale, camera_dict = readNerfiesCameras_mv(path, load_time_step=load_time_step)

    train_cam_infos = cam_infos[:train_num]
    test_cam_infos = cam_infos[train_num:]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    ply_path = os.path.join(path, "duster_points3d.ply")
    _mesh = trimesh.load(ply_path)
    xyz = np.asarray(_mesh.vertices)
    # subsample points
    if max_pts > 0 and xyz.shape[0] > max_pts:
        xyz = xyz[np.random.choice(xyz.shape[0], max_pts, replace=False)]
    xyz = (xyz - scene_center) * scene_scale
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(np.random.random((xyz.shape[0], 3)) / 255.0), normals=np.zeros_like(xyz))
    # if not os.path.exists(ply_path):
    #     print(f"Generating point cloud from nerfies...")

    #     xyz = np.load(os.path.join(path, "points.npy"))
    #     xyz = (xyz - scene_center) * scene_scale
    #     num_pts = xyz.shape[0]
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
    #         shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    vis_C2W = []
    # static_train_cam = [cam for cam in train_cam_infos if cam.time == 0]
    vis_cam_order = [0, 4, 12, 5, 3, 10, 8, 11, 9, 2, 7, 6] # for 4x downsampling
    vis_cam_order = [10, 6, 8, 12, 7, 3, 0, 9, 2, 5, 4, 11] + [10, 6]
    cam_id_order = [camera_dict[vis_cam_id] for vis_cam_id in vis_cam_order]
    for cam in cam_id_order:
        # W2C = getWorld2View2(cam.R, cam.T)
        # C2W = np.linalg.inv(W2C)
        # vis_C2W.append(C2W)
        Rt = np.eye(4)
        Rt[:3, :3] = cam.R
        Rt[:3, 3] = cam.T
        vis_C2W.append(np.linalg.inv(Rt))
    vis_C2W = np.stack(vis_C2W)[:, :3, :4]
    # interpolate between cameras
    visualization_poses = generate_interpolated_path(vis_C2W, 50, spline_degree=3, smoothness=0.0, rot_weight=0.01)
    video_cameras = []
    video_cam_centers = []
    for _idx, _pose in enumerate(visualization_poses):
        Rt = np.eye(4)
        Rt[:3, :4] = _pose[:3, :4]
        Rt = np.linalg.inv(Rt)
        R = Rt[:3, :3]
        T = Rt[:3, 3]
        video_cameras.append(CameraInfo(
                uid=_idx, fid=0, #   fid=fid, time=fid,
                R=R, T=T,
                FovY=train_cam_infos[0].FovY, FovX=train_cam_infos[0].FovX,
                image=None, image_path=None, image_name=f"{_idx:06}", 
                width=train_cam_infos[0].image.size[0], height=train_cam_infos[0].image.size[1],
        ))
        video_cam_centers.append(_pose[:3, 3])
    if False:
        # tmp save point cloud for debugging
        import trimesh 
        # import pyrender
        # scene = pyrender.Scene()
        video_cam_centers = np.stack(video_cam_centers, axis=0)
        stat_cam_centers = vis_C2W[:, :3, 3]
        _scene = []
        _scene.append(vis_create_pc(video_cam_centers, color=(1.0, 0.0, 0.0), radius=0.005))
        _scene.append(vis_create_pc(stat_cam_centers, color=(0.0, 1.0, 0.0), radius=0.005))
        _scene.append(trimesh.load(ply_path))
        # scene.add(vis_create_pc(_scene_pc.vertices, color=(0.5, 0.5, 0.5), radius=0.003))
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        # trimesh union of point clouds
        # merge meshes 
        all_vertices = np.concatenate([_m.vertices for _m in _scene], axis=0)
        all_colors = np.concatenate([_m.visual.vertex_colors for _m in _scene], axis=0)
        _ = trimesh.PointCloud(all_vertices, colors=all_colors).export('tmp_nerfies.ply')
        print('Saved tmp_nerfies.ply')

        # video_point_cloud = np.concatenate([video_cam_centers, stat_cam_centers], axis=0)
        # video_point_cloud_colors = np.concatenate([np.ones((video_cam_centers.shape[0], 3)), np.zeros((stat_cam_centers.shape[0], 3))], axis=0)
        # _pc = trimesh.PointCloud(video_point_cloud, video_point_cloud_colors)
        # _scene_pc = trimesh.load(ply_path)
        # _scene_pc.vertices = np.concatenate([_scene_pc.vertices, video_point_cloud], axis=0)
        # _scene_pc.visual.vertex_colors = np.concatenate([_scene_pc.visual.vertex_colors, video_point_cloud_colors], axis=0)
        # _scene_pc.export('tmp_nerfies.ply')
        # _pc.export('tmp_nerfies.ply')
        breakpoint()


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pred_cameras=video_cameras,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Colmap": readColmapSceneInfoSparse,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "Blender_cv": readNerfSyntheticCVInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo_mv,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "ResFields": readResFieldSceneInfo,
}
