# SISA3D

## Aligning the Gaussians with the Surface
The primary goal of the SISA3D framework is to, following the SuGaR framework, adding regularization terms to the
DreamGaussian framework to encourage the Gaussians in the scene to take on properties that should faciliate better 
mesh extraction. Specifically, we can use the DreamGaussian framework transform a single 2D image into a (coarse) 3D Gaussian Splatting Scene in the form of a .ply file. The remaining challenge  is to (1) optimize the scene in this .ply file further such that the Gaussians are better aligned with the surface of the object we try to model, and (2) to then to extract a mesh object from this refined 3DGS scene in the form of a .obj file. This repository contains the source code related to three independent approaches to do so:


### 1. Using SuGaR
In theory, [SuGaR](https://github.com/Anttwo/SuGaR) can be applied to any 3D Gaussian Splatting Scene (.ply file) that has been optimized for 7K iterations. However, SuGaR was developed to extract meshes from traditional 3DGS scenes that are optimized by transforming a large set of 2D images from different angles into a COLMAP dataset (using Structure from Motion (SfM)) and using 
that 3D information to guide the optimizaiton process. And even the surface alignment stage still seems to make use of the COLMAP dataset to guide the optimizaiton process, as shown by this command from SuGaR's README:

```bash
# taken from SuGar's README:
python train.py -s <path to COLMAP dataset> -c <path to the Gaussian Splatting checkpoint> -r <"density" or "sdf">
```

However, since we want to transform a single 2D image into a 3D object and thus do not have a large set of 2D images available
to create a COLMAP dataset, applying SuGaR to our generated 3DGs scene is not trivial. 

One approach we tried is to create a coarse 3DGS scene from our 2D image, and create an artificial set of 2D images of that object by rendering that 3D model onto the 2D plane from different (informed) point of views. However, in our experiments, the traditional SfM algorithm (that is also used by the original 3DGS paper) does not converge on such artificially created images, likely because the images are too blurry (which is exactly why we want to apply SuGaR in the first place.)

To reproduce the results for this experiment, first install the colmap environment and then run the corresponding jobs:

```bash
sbatch jobs/install_colmap_environment.job
sbatch jobs/create_colmap_dataset.job
```

### 2. Using our own regularization terms.
As of now, this lead to CLIP scores slightly below baseline performange of DreamGaussian (.69 vs .70).
The source code for this approach can be found under *ptdraft/sisa3d/*

This approach works, but procuces slightly worse clip scores than baseline (DreamGaussian) performance.
See the *Job Scripts and Associated Configuration Files* section for more information.

### 3. Using the (Coarse) Sugar stage of the MVControl framework
The [MVControl framework](https://github.com/WU-CVGL/MVControl-threestudio) seems to succesfully use SuGaR in a generative setting (but in a text-to-3D setting, rather than a single-image-to-3D setting as we try in SISA3D). In their README,
they point to a function that seems to imply they developed a function *extract_mesh.py* that can apply SuGaR and extract
a mesh from a optimized 3DGS scene (.ply file):

```bash
# Taken from the MVControl README
# ...
refined_gs_path=$exp_root_dir/gaussian_refine@LAST/save/exported_gs_step3000.ply
coarse_sugar_output_dir=$exp_root_dir/coarse_sugar
python extern/sugar/extract_mesh.py -s extern/sugar/load/scene \
-c $refined_gs_path -o $coarse_sugar_output_dir --use_vanilla_3dgs
```

However, once running this script, we find it expects to find a *config/parsed.yaml" path to be present 
in the checkpoint directory (-c argument). Finding out what are the contents of this parsed.yaml and if we can
create our own is our current line of investigation.

To continue on this path, consider installing the corresponding environment and reproducing the error by running the *run_mvc.job*:

```bash
# install the environment if you haven't done so already:
sbatch install_mvcontrol_environment
```
```bash
# run the mvc job to reproduce the error
sbatch run_sugar.job
```



## Job Scripts and Associated Configuration Files for Approach 2: Using Our Own Regularization Terms:

> [!NOTE]
> All the following jobs require the DreamGaussianV2 environment. If you haven't done so already, install it before running the job:
> ```bash
> sbatch jobs/install_dream_gaussian_environment.job
>```

### run_pipeline.job

> [!CAUTION]
> This job is currently broken. Will need to fix it after finding a configuration that works on par or better than baseline.

Transforms a single image into a 3D Gaussian splatting scene (.ply file) and optionally into a mesh (.obj) file
if the second stage of the pipeline is actived in the configuration file. In addtion, after each stage, the pipeline computes valuation metrics (clip scores for quantivative evaluation; images for qualitative evaluation) and stores them in 
the .csv files / output directories specified in the configuration file.

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `config.yaml`               | Defines the configuration for a single run.| |


#### Running the job:

```bash
sbatch run_pipline.job
```

### run_gridsearch.job
Runs stage 1 for different configurations defined in the grid_search.yaml configuration file and saves the average clip
score for each hyperparameter setting in a predefined .csv file. This .csv can then be read in the *grid_search.ipynb"
notebook to retrieve the configuration with the best clip score.

To faciliate doing a grid search over only a (small) subset of all hyperparameters, this job uses two configuration files: any array provided in *grid_search.yaml* will be used as a range to do a grid 
search over; for parameters that are not defined in the *grid_search.yaml* file, the job will fall back to the values
defined in the *config.yaml* file. 

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `config.yaml` | Defines the default configuration to fall back on for paramters not defined in grid_search.yaml | |
| `grid_search.yaml`       | Defines the range of values to do grid search over for a subset of the possible hyperparameters| All values (even single values) should be placed in a list here, for otherwise the script will fail. |

#### Running the job:
```bash
sbatch run_gridsearch.job 
```
#### Results
Aftering running the job, a .csv file with clip scores for each hyperparameter configuration will be produced. 
You can read and analyze this .csv file with grid_search.ipynb to determine which configuration produces the best results.

### run_compare.job
Runs two full pipelines with different configurations to produce results usable for in a thesis / paper.

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `compare_config.woreg.yaml` | Defines the configuration for the baseline model (without SuGaR regularization)|
| `compare_config.yaml`       | Defines the configuration for our model (with SuGaR regularization)   | Configuration will likely be updated as we find better hyperparameter configurations, etc. |


#### Running the job:
```bash
sbatch run_compare.job
```

#### Results
The job will produce the following results:

- a .csv file with the clip scores for each config
- a directory with .png files with the images generated by the different configurations from different angles.

These files can be further procesed in *results.ipynb* to produce figures and tables.

## Job Scripts and Associated Configuration Files for Approach 3: Using MVControl's SuGaR Stage:
> [!NOTE]
> The following jobs require the **mvcontroljune6** environment. If you haven't done so already, install it before running the job:
> ```bash
> sbatch install_mvcontrol_environment.job
>```


### run_sugar.job
Tries to run the SuGaR stage of the MVControl framework on a .ply file generated by the DreamGaussian framework.
However, this fails because we do not have a "config/parsed.yaml" in the directory containing our .ply file.

#### Running the job:

```bash
sbatc run_sugar.job
```

#### Results
will produce the error:

```bash
Traceback (most recent call last):
  File "extern/sugar/extract_mesh.py", line 66, in <module>
    cfg = load_config(cfg_path)
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/MVControl-threestudio/./threestudio/utils/config.py", line 108, in load_config
    yaml_confs = [OmegaConf.load(f) for f in yamls]
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/MVControl-threestudio/./threestudio/utils/config.py", line 108, in <listcomp>
    yaml_confs = [OmegaConf.load(f) for f in yamls]
  File "/home/jhuizing/.conda/envs/mvcontroljune6/lib/python3.8/site-packages/omegaconf/omegaconf.py", line 189, in load
    with io.open(os.path.abspath(file_), "r", encoding="utf-8") as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/jhuizing/master-thesis-ds/results/stage_1/configs/parsed.yaml'

```



