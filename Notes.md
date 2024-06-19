# Master Thesis

images taken from: https://github.com/lukemelas/realfusion
## Visualize .ply files online:

https://imagetostl.com/view-ply-online#convert

mp4 to sequence of images:

https://ezgif.com/video-to-jpg/ezgif-1-86167c97b4.mp4

# Snellius 
## Connecting to Snellius:

```bash
ssh -X jhuizing@snellius.surf.nl
```

## Set-up
Ensure you clone the repository recursively such that all submodules get loadeded correctly:

> [!CAUTION]
> Ideally we would use an environment.yaml file to take care of our dependencies. Unfortunately, DreamGaussian makes use
> of submodules, which cannot be installed via conda. So we do need to use pip directly.

> [!WARNING]
> Consider removing the repo altogether before cloning in; installing depenendecies from submodules can fail otherwise.


> [!WARNING]
> Although **git submodule** add adds the submodule, it does not automatically load nested submodules. 
> To do so, run
> **git submodule update --init --recursive** inside the submodule.


> [!WARNING]
> The rasterizatin submodule seems to be causing a lot of dependency problems...
> https://chat.openai.com/share/3166f123-b5dd-47ac-a548-4d20fd1f6290
```bash
git clone --recursive https://github.com/JuliusHuizing/master-thesis-ds
cd master-thesis-ds
sbatch install_environment.job
```





```
## Jobs

- Running a job:
```bash
sbatch JOBNAME
```
- Listing all job stati
```bash
squeue
```

- Cancel a job:
```bash
scancel JOBID
```
- Show additional information of a specific job, like the estimated start time.
```bash
scontrol show job JOBID
```

## echo errors:
```bash
awk 'tolower($0) ~ /error/ {print; err=1; next} /^[ \t]/ && err {print; next} {err=0}' filename

```

## References:
- https://servicedesk.surf.nl/wiki/display/WIKI/Software+policy+Snellius#SoftwarepolicySnellius-UseofAnacondaandMinicondaenvironmentsonSnellius
- https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi

# Working with Git Submodules
- https://github.blog/2016-02-01-working-with-submodules/

> [!NOTE]  
> Highlights information that users should take into account, even when skimming.

> [!TIP]
> Optional information to help a user be more successful.

> [!IMPORTANT]  
> Crucial information necessary for users to succeed.

> [!WARNING]  
> Critical content demanding immediate user attention due to potential risks.

> [!CAUTION]
> Negative potential consequences of an action.




## Creating a Colmap Dataset
The API of SuGaR requires a colmap dataset.
We can use the convert.py script of the original 3DGS paper to create a Colmap dataset from a collection of images, but these images need to: 

- [X] have the same resolution
  - For this, we can simply delete the lower res images
- [ ] be of sufficient resolution and have enough overlap for the algorithm to find a good initial pair, otherwise you'll get the error:

```error
Finding good initial image pair
==============================================================================
I20240429 12:16:33.909912 22618458595328 incremental_mapper.cc:404] => No good initial image pair found.
I20240429 12:16:33.909923 22618458595328 timer.cc:91] Elapsed time: 0.000 [minutes]
E20240429 12:16:33.912605 22620776206336 sfm.cc:266] failed to create sparse model
ERROR:root:Mapper failed with code 256. Exiting.

```



# Creating A Colmap Dataset instructions
https://colmap.github.io/tutorial.html


# increase Git PostBuffer to allow for larger pushes
https://medium.com/swlh/everything-you-need-to-know-to-resolve-the-git-push-rpc-error-1a865fd1ebea


```bash
git config http.postBuffer 2147483648
```




COLAMP:

https://colmap.github.io/tutorial.html :

If you have control over the picture capture process, please follow these guidelines for optimal reconstruction results:

Capture images with good texture. Avoid completely texture-less images (e.g., a white wall or empty desk). If the scene does not contain enough texture itself, you could place additional background objects, such as posters, etc.
Capture images at similar illumination conditions. Avoid high dynamic range scenes (e.g., pictures against the sun with shadows or pictures through doors/windows). Avoid specularities on shiny surfaces.
Capture images with high visual overlap. Make sure that each object is seen in at least 3 images – the more images the better.
Capture images from different viewpoints. Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot. At the same time, try to have enough images from a relatively similar viewpoint. Note that more images is not necessarily better and might lead to a slow reconstruction process. If you use a video as input, consider down-sampling the frame rate.

# debugger while on Snellius

> [!NOTE]  
> If you cannot set breakpoints in a file, try reloading the window in vscode (cmd + shift + p; "> reload window")

1. locally: set up ~/.shh/config
2. on snellius: spin up sleep.job
3. when sleep job is running, run squeue to see the node; 
  locally, in ~/.shh/config, replace node name with target node in 
4. on VSCODe, connect to snellius proxy (blue lower left button.)
5. activate environment (source activate sugar) in terminal connected to snellius proxy
6. find the main file you want to run (e.g. train.py)
7. with the file open, start the debugger (left column bar, debug extension)
8. Provide arguments in prompt if necessary; press enter
9. debugger will run and output should show in terminal.
10. WHen done, do not forget to scancel the sleep job.




# Transfer large files from snellius to local

locally, e.g. run:

```bash
REMOTE_PATH="/home/jhuizing/master-thesis-ds/repos/SuGaR/output/coarse_mesh/colmap/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply"
rsync -av jhuizing@snellius.surf.nl:$REMOTE_PATH .
```

https://opensfm.org/docs/geometry.html?highlight=brown#brown-camera
https://community.opendronemap.org/t/cameras-json-format-documentation/19162


we have this camera_to_json method:

```python
def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

```

and we use this to generate images correctly:

```python
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

```
However, we thus have a MiniCam rather than a Camera Object, which the camera_to_json requires...

```python

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


```

see repos/gaussian-splatting/scene/cameras.py

https://www.overleaf.com/learn/latex/Positioning_images_and_tables