import os
import numpy as np
import cv2

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

def capture_and_save_images(camera_positions, directory, step, ref_size, fovy, fovx, near, far, renderer, orbit_camera, MiniCam):
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
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(directory, f'rendered_image_{step}_{idx}.jpg'), image_np)

