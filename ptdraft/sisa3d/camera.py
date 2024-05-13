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
    create_directory(directory)
    for idx, (ver, hor, rad) in enumerate(camera_positions):
        pose = orbit_camera(ver, hor, rad)
        cur_cam = MiniCam(pose, ref_size, ref_size, fovy, fovx, near, far)
        out = renderer.render(cur_cam)
        image = out["image"].unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # create directory if not exists
        name = f'v{ver}_h{hor}_r{rad}_s{step}_i{idx}.jpg'
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
        name = f'v{ver}_h{hor}_r{rad}_s{step}_i{idx}.jpg'
        cv2.imwrite(os.path.join(generated_output_path, name), image_np)
        
        # save reference png
        convert_png_to_jpeg(path_to_preproccesed_reference_image, os.path.join(reference_output_path, name), jpeg_quality=100)

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


def convert_png_to_jpeg(input_png_path, output_jpeg_path, jpeg_quality=95):
    """
    Converts an image from PNG format to JPEG format.

    Args:
        input_png_path (str): Path to the input PNG image file.
        output_jpeg_path (str): Path where the output JPEG image will be saved.
        jpeg_quality (int): Quality of the output JPEG image, ranging from 0 (worst) to 100 (best). Default is 95.

    Returns:
        bool: True if the image was successfully converted and saved, False otherwise.
        
    
    Example usage:
    
        success = convert_png_to_jpeg("input.png", "output.jpg", 95)

    """
    # Read the image from the specified path
    image = cv2.imread(input_png_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded properly
    if image is None:
        print("Error: Image not loaded. Please check the file path.")
        return False

    # Write the image to a new JPEG file with specified quality
    result = cv2.imwrite(output_jpeg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    if result:
        print("Image successfully converted to JPEG and saved at:", output_jpeg_path)
        return True
    else:
        print("Failed to write the JPEG image.")
        return False
