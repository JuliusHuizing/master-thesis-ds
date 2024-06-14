
We can now create a mesh object using dreamgaussian and sugar by first running a dream gaussian job that saves camera information:

```bash
sbatch run run_dreamgaussian_with_camera_information.job
```

This will produce in the `dg_for_sugar` dir:
- a .ply file (point cloud)
- images of the point cloud
- camera information associated with the imagse (cameras.json)

We can now run a sugar job on this output to refine the point cloud and create a mesh:

```bash
sbatch run_sugar.job
```

This will create a mesh object under `repos/SuGaR/output`


Nevertheless, the resulting mesh looks very bad and much worse the dream gaussian output (note the mesh file is to large in size to put under version control, but we can use the `rsync` command to get and inspect it locally). We can probably improve the quality by ensuring we provide sugar with better images and camera information of the point cloud. However, giving it images with a full 360 degree coverage of the point cloud causes the SuGaR algortihm to not converge at all....


The relevant code capturing these images and camera information can be found in dreamgaussians' main.py and our own camera package. The images we generate of the point cloud seem good, but perhaps I a doing something wrong with saving the associated camera infomation...

- `repos/dreamgaussian/main.py'  (line 549)

```python
        if self.opt.save_camera_positions:
            
            # TODO: this approach does not result in a valid sugar .obj...
            # camera_positions = self.sample_camera_positions_for_sugar()
            # file_name = self.opt.input.split("/")[-1].split(".")[0] # hack, we dont need this
            # capture_and_save_images_for_sugar(file_name, camera_positions, self.opt.stage_1_result_images_output_path, self.step, self.opt.ref_size, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, self.renderer, orbit_camera, MiniCam)
            # self.save_camera_ply()
            
            # This, for some reason, does... although we use worse camera data 
            camera_positions = self.sample_random_camera_positions(100)
            image_names = self.save_images_for_camera_positions(camera_positions)
            self.save_camera_information(camera_positions, image_names)
            self.save_camera_ply()
```

- `ptdraft/sisa3d/camera.py` (line 54)

```python
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


```