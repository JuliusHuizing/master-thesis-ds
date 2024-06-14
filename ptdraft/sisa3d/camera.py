import os
import numpy as np
import cv2
from PIL import Image
import uuid
import json

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
import numpy as np
from typing import NamedTuple
from torch import nn
 

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def create_directory(directory):
    """
    Create a directory if it does not exist.
    
    Args:
        directory (str): The directory path to create.
    """
    print("[INFO] Creating directory for images!")
    os.makedirs(directory, exist_ok=True)

def generate_camera_positions(num_positions, elevation, radius):
    """
    Generate camera positions with overlap and varied viewpoints.
    
    Args:
        num_positions (int): Number of camera positions to generate.
        elevation (int): Base elevation for camera positions.
        radius (float): Base radius for camera positions.
    
    Returns:
        list of tuples: Each tuple contains (elevation, horizontal angle, radius).
    """
    positions = []
    step_angle = 360 / num_positions
    for i in range(num_positions):
        ver = np.random.randint(-30, 30)
        hor = step_angle * i
        for radius_variation in [0.8, 0.9, 1.0, 1.1, 1.2]:
            positions.append((elevation + ver, hor, radius * radius_variation))
    return positions


    #   save_dir = "dg_for_sugar/colmap/images"
    #     os.makedirs(save_dir, exist_ok=True)
    #     image_names = []
    #     for cam in camera_positions:
    #         pose = orbit_camera(*cam['position'], self.opt.radius)
    #         cur_cam = MiniCam(pose, self.W, self.H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
    #         out = self.renderer.render(cur_cam)
    #         image = out["image"].permute(1, 2, 0).cpu().detach().numpy() * 255
    #         img_name = f"image_{cam['id']}.png"
    #         cv2.imwrite(os.path.join(save_dir, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #         image_names.append(img_name)
    #     return image_names

def capture_and_save_images_for_sugar(image_name, camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
    """
    Captures and saves images based on given camera positions using the specified camera and rendering settings.
    
    Args:
        image_name (str): Name of the image.
        camera_positions (list): Camera positions (elevation, horizontal angle, radius).
        directory (str): Directory to save images.
        step (int): Current step or iteration in the process.
        ref_size (int): Reference size for the camera.
        fovy (float): Vertical field of view.
        fovx (float): Horizontal field of view.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        renderer: Rendering engine.
        orbit_camera (function): Function to compute the camera pose based on positions.
        MiniCam (class): Camera class for initializing camera settings.
    """
    
    cameras_data = []
    save_dir = "dg_for_sugar/colmap/images"
    os.makedirs(save_dir, exist_ok=True)
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        name = f"image_{idx}.png"
        image_path = os.path.join(save_dir, name)
        cv2.imwrite(image_path, image_np)

        # Extract width and height from the image
        height, width, _ = image_np.shape

        # Extract position and rotation from the pose matrix
        position = pose[:3, 3]
        rotation = pose[:3, :3]

        # Create dictionary for the current image
        nam_without_extension = f"image_{idx}"
        camera_dict = {
            "id": idx,
            "img_name": nam_without_extension,
            "width": width,
            "height": height,
            "position": position.tolist(),
            "rotation": rotation.tolist(),
            "fy": fovy,
            "fx": fovx
        }

        cameras_data.append(camera_dict)

    # Save cameras_data to cameras.json
    save_dir = "dg_for_sugar/checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cameras.json"), 'w') as json_file:
        json.dump(cameras_data, json_file, indent=4)
        
def capture_and_save_images_for_sugarV2(image_name, camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
    """
    Captures and saves images based on given camera positions using the specified camera and rendering settings.
    
    Args:
        image_name (str): Name of the image.
        camera_positions (list): Camera positions (elevation, horizontal angle, radius).
        directory (str): Directory to save images.
        step (int): Current step or iteration in the process.
        ref_size (int): Reference size for the camera.
        fovy (float): Vertical field of view.
        fovx (float): Horizontal field of view.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        renderer: Rendering engine.
        orbit_camera (function): Function to compute the camera pose based on positions.
        MiniCam (class): Camera class for initializing camera settings.
    """
    
    cameras_data = []
    save_dir = "dg_for_sugar/colmap/images"
    os.makedirs(save_dir, exist_ok=True)
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        name = f"image_{idx}.png"
        image_path = os.path.join(save_dir, name)
        cv2.imwrite(image_path, image_np)

        # Extract width and height from the image
        height, width, _ = image_np.shape

        # Extract position and rotation from the pose matrix
        position = spherical_to_cartesian(rad, hor, ver)
        rotation = calculate_rotation_matrix(hor, ver)

        # Create dictionary for the current image
        nam_without_extension = f"image_{idx}"
        camera_dict = {
            "id": idx,
            "img_name": nam_without_extension,
            "width": width,
            "height": height,
            "position": position.tolist(),
            "rotation": rotation.tolist(),
            "fy": fovy,
            "fx": fovx
        }

        cameras_data.append(camera_dict)

    # Save cameras_data to cameras.json
    save_dir = "dg_for_sugar/checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cameras.json"), 'w') as json_file:
        json.dump(cameras_data, json_file, indent=4)


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    y = radius * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
    z = radius * np.sin(np.radians(elevation))
    return np.array([x, y, z])

def calculate_rotation_matrix(azimuth, elevation, roll=0):
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    roll = np.radians(roll)
    
    # Rotation matrix for azimuth (yaw)
    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix for elevation (pitch)
    Ry = np.array([
        [np.cos(elevation), 0, np.sin(elevation)],
        [0, 1, 0],
        [-np.sin(elevation), 0, np.cos(elevation)]
    ])
    
    # Rotation matrix for roll
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def capture_and_save_images(image_name, camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
    """
    Captures and saves images based on given camera positions using the specified camera and rendering settings.
    
    Args:
        image_name (str): Name of the image.
        camera_positions (list): Camera positions (elevation, horizontal angle, radius).
        directory (str): Directory to save images.
        step (int): Current step or iteration in the process.
        ref_size (int): Reference size for the camera.
        fovy (float): Vertical field of view.
        fovx (float): Horizontal field of view.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        renderer: Rendering engine.
        orbit_camera (function): Function to compute the camera pose based on positions.
        MiniCam (class): Camera class for initializing camera settings.
    """
    create_directory(directory)
    # touch a tmp file in this dirimport os

    # tmp_file_path = os.path.join(directory, "tmp.txt")
    # with open(tmp_file_path, 'a'):
    #     os.utime(tmp_file_path, None)

    # Call the function with the directory path
  
    
    
    # create a file called tmp.txt in the directory
    # with open(os.path.join(directory, "tmp.txt"), "w") as f:
    #     f.write("This is a temporary file.")
    print("[INFO] Capturing and saving images in directory: ", directory, flush=True)
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        tmp_file_path = os.path.join(directory, f"tmp_{idx}.txt")
        # with open(tmp_file_path, 'a'):
        #     os.utime(tmp_file_path, None)
        # create a file called tmp_idx.txt in the directory
        # with open(os.path.join(directory, f"tmp_{idx}.txt"), "w") as f:
        #     f.write(f"This is a temporary file for idx {idx}.")
        print("[INFO] Capturing and saving one image in directory: ", directory, flush=True)
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        unique_id = uuid.uuid4().hex
        # DO NOT USE THE NAME VARIABLE in the name, for then images won't be saved...
        name = f'{image_name}_capture_and_save_v{ver}_h{hor}_r{rad}_s{step}_i{idx}_uuid_{unique_id}.jpg'
        cv2.imwrite(os.path.join(directory, name), image_np)
        
        
def capture_and_save_images_for_clip_similarity(path_to_preproccesed_reference_image, camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
    """
    Captures and saves images based on given camera positions using the specified camera and rendering settings.
    
    Args:
        camera_positions (list): Camera positions (elevation, horizontal angle, radius).
        directory (str): Directory to save images.
        step (int): Current step or iteration in the process.
        ref_size (int): Reference size for the camera.
        fovy (float): Vertical field of view.
        fovx (float): Horizontal field of view.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        renderer: Rendering engine.
        orbit_camera (function): Function to compute the camera pose based on positions.
        MiniCam (class): Camera class for initializing camera settings.
    """
    reference_output_path = os.path.join(directory, "reference")
    generated_output_path = os.path.join(directory, "generated")
    create_directory(reference_output_path)
    create_directory(generated_output_path)
    

    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # create directory if not exists
        unique_id = uuid.uuid4().hex
        name = f'v{ver}_h{hor}_r{rad}_s{step}_i{idx}_uuid_{unique_id}.jpg'
        cv2.imwrite(os.path.join(generated_output_path, name), image_np)
        
        # save reference png

        im = Image.open(path_to_preproccesed_reference_image)
        rgb_im = im.convert('RGB')
        rgb_im.save(os.path.join(reference_output_path, name))

def generate_fixed_elevation_positions(azimuth_angles, elevation, radius):
    """
    Generate camera positions at a fixed elevation and specified azimuth angles.

    Args:
        azimuth_angles (list of int): List of azimuth angles.
        elevation (int): Elevation angle, set to zero for this case.
        radius (float): Distance from the origin.

    Returns:
        list of tuples: Each tuple contains (elevation, azimuth angle, radius).
    """
    return [(elevation, angle, radius) for angle in azimuth_angles]


def save_camera_information(self, camera_positions, image_names):
    camera_data = []
    for cam, img_name in zip(camera_positions, image_names):
        cam_info = {
            'id': cam['id'],
            'img_name': img_name,
            'width': self.W,
            'height': self.H,
            'position': cam['position'],
            'rotation': cam['rotation'],
            'fx': self.cam.fovx,
            'fy': self.cam.fovy
        }
        camera_data.append(cam_info)
    with open('cameras.json', 'w') as f:
        json.dump(camera_data, f, indent=4)
        
        
def save_images_for_camera_positions(self, camera_positions):
    save_dir = "generated_images"
    os.makedirs(save_dir, exist_ok=True)
    image_names = []
    for cam in camera_positions:
        pose = orbit_camera(*cam['position'], self.opt.radius)
        cur_cam = MiniCam(pose, self.W, self.H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
        out = self.renderer.render(cur_cam)
        image = out["image"].permute(1, 2, 0).cpu().detach().numpy() * 255
        img_name = f"image_{cam['id']}.png"
        cv2.imwrite(os.path.join(save_dir, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_names.append(img_name)
    return image_names

# # taken from 3dgs lib
# def camera_to_JSON(id, camera : Camera):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = camera.R.transpose()
#     Rt[:3, 3] = camera.T
#     Rt[3, 3] = 1.0

#     W2C = np.linalg.inv(Rt)
#     pos = W2C[:3, 3]
#     rot = W2C[:3, :3]
#     serializable_array_2d = [x.tolist() for x in rot]
#     camera_entry = {
#         'id' : id,
#         'img_name' : camera.image_name,
#         'width' : camera.width,
#         'height' : camera.height,
#         'position': pos.tolist(),
#         'rotation': serializable_array_2d,
#         'fy' : fov2focal(camera.FovY, camera.height),
#         'fx' : fov2focal(camera.FovX, camera.width)
#     }
#     return camera_entry

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


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def minicam_to_JSON(id, camera: MiniCam):
    # Inverting the world view transform to get the camera position and rotation
    world_view_transform_inv = torch.inverse(camera.world_view_transform)
    camera_center = world_view_transform_inv[3][:3].numpy()
    rot = world_view_transform_inv[:3, :3].numpy()

    # Constructing the camera entry dictionary
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'width': camera.image_width,
        'height': camera.image_height,
        'position': camera_center.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FoVy, camera.image_height),
        'fx': fov2focal(camera.FoVx, camera.image_width)
    }
    return camera_entry


def capture_and_save_images_for_sugarV3(image_name, camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
    """
    Captures and saves images based on given camera positions using the specified camera and rendering settings.
    
    Args:
        image_name (str): Name of the image.
        camera_positions (list): Camera positions (elevation, horizontal angle, radius).
        directory (str): Directory to save images.
        step (int): Current step or iteration in the process.
        ref_size (int): Reference size for the camera.
        fovy (float): Vertical field of view.
        fovx (float): Horizontal field of view.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        renderer: Rendering engine.
        orbit_camera (function): Function to compute the camera pose based on positions.
        MiniCam (class): Camera class for initializing camera settings.
    """
    
    cameras_data = []
    save_dir = "dg_for_sugar/colmap/images"
    os.makedirs(save_dir, exist_ok=True)
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        name = f"image_{idx}.png"
        image_path = os.path.join(save_dir, name)
        cv2.imwrite(image_path, image_np)

        # Extract width and height from the image
        camera_dict = minicam_to_JSON(idx, cur_cam)
        cameras_data.append(camera_dict)

    # Save cameras_data to cameras.json
    save_dir = "dg_for_sugar/checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cameras.json"), 'w') as json_file:
        json.dump(cameras_data, json_file, indent=4)