import os
import numpy as np
import cv2
from PIL import Image
import uuid


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
        name = f'capture_and_save_v{ver}_h{hor}_r{rad}_s{step}_i{idx}_uuid_{unique_id}.jpg'
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



