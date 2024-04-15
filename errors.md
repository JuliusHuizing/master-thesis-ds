   RuntimeError:
      The detected CUDA version (11.8) mismatches the version that was used to compile
      PyTorch (12.1). Please make sure to use the same CUDA versions.


ERROR: Could not find a version that satisfies the requirement opencv (from versions: none)
ERROR: No matching distribution found for opencv
WARNING: There was an error checking the latest version of pip.



when settigng python to 3.7, installation succeeds; but run fails; python 3.7 does not have Literal from Typing.
When settign python to 3.8 (which supports Literal), installation fails:

× python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [53 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.linux-x86_64-cpython-38
      creating build/lib.linux-x86_64-cpython-38/diff_gaussian_rasterization
      copying diff_gaussian_rasterization/__init__.py -> build/lib.linux-x86_64-cpython-38/diff_gaussian_rasterization
      running build_ext
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/gpfs/home6/jhuizing/master-thesis-ds/repos/dreamgaussian/diff-gaussian-rasterization/setup.py", line 17, in <module>
          setup(
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/__init__.py", line 103, in setup
          return distutils.core.setup(**attrs)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 185, in setup
          return run_commands(dist)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
          dist.run_commands()
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/dist.py", line 989, in run_command
          super().run_command(command)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/wheel/bdist_wheel.py", line 364, in run
          self.run_command("build")
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/dist.py", line 989, in run_command
          super().run_command(command)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/command/build.py", line 131, in run
          self.run_command(cmd_name)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/dist.py", line 989, in run_command
          super().run_command(command)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 88, in run
          _build_ext.run(self)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 345, in run
          self.build_extensions()
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 523, in build_extensions
          _check_cuda_version(compiler_name, compiler_version)
        File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 414, in _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (11.6) mismatches the version that was used to compile
      PyTorch (12.1). Please make sure to use the same CUDA versions.






using cuda 12.1.1 (as 3dgs repo suggests) let dream gaussian setup fail altogether.


using cuda 12.1.1 with python 3.8 will complain about cuda version not set during env install.

note that the nightly of cuda 11.6 is indeed deprecated, explaining previous error when trying to install
that one: https://pytorch.org/blog/deprecation-cuda-python-support/

errors:

https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix


when not specifying python version, installation process will default to 3.12, causing this error:
https://stackoverflow.com/questions/77364550/attributeerror-module-pkgutil-has-no-attribute-impimporter-did-you-mean

specifying python 3.9 cannot resolve dependencies for specified torch version...

python 3.8 cannot find matching scipy version.

https://github.com/dreamgaussian/dreamgaussian/issues/



this works:

```bash
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

module purge
# module load 2021 // although spider says we need 2021 for cuda 11.6, the partition does not support 2021..
module load 2022
module load CUDA/11.8.0
module load Anaconda3/2022.05

cd $HOME/master-thesis-ds/
git pull

cd $HOME/master-thesis-ds/repos/dreamgaussian
# conda env remove --name dreamgaussian
conda create -n dreamgaussian python=3.8 pip
source activate dreamgaussian

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream


```


But now running the training loop gives an error, indicating the linux version we use is too old..
Which we cannot easily update.

```error
Traceback (most recent call last):
  File "main.py", line 6, in <module>
    import dearpygui.dearpygui as dpg
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/dearpygui/dearpygui.py", line 22, in <module>
    import dearpygui._dearpygui as internal_dpg
ImportError: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/dearpygui/_dearpygui.so)
Traceback (most recent call last):
  File "main2.py", line 6, in <module>
    import dearpygui.dearpygui as dpg
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/dearpygui/dearpygui.py", line 22, in <module>
    import dearpygui._dearpygui as internal_dpg
ImportError: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/dearpygui/_dearpygui.so)
Traceback (most recent call last):
  File "/home/jhuizing/.conda/envs/dreamgaussian/bin/kire", line 8, in <module>
    sys.exit(main())
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/kiui/render.py", line 431, in main
    gui = GUI(opt)
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/kiui/render.py", line 44, in __init__
    self.mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir)
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/kiui/mesh.py", line 96, in load
    mesh = cls.load_obj(path, **kwargs)
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/kiui/mesh.py", line 182, in load_obj
    with open(path, "r") as f:
FileNotFoundError: [Errno 2] N
```


Stackoverflow suggests this can be solved by using conda-forage... but
When trying to isntall via conda environment, solving environment takes forever:
https://stackoverflow.com/questions/63734508/stuck-at-solving-environment-on-anaconda


We could actually solve it by removing all references to dearpygui in main.py (main2.py still todo);
It is just likely that a non-gui system like snellius us running a disto of ubuntu that does not support 
dearpy gui dependencies.


Now we run into: 
```error:
Traceback (most recent call last):
  File "main.py", line 616, in <module>
    gui.train(opt.iters)
  File "main.py", line 597, in train
    self.save_model(mode='geo+tex')
  File "/home/jhuizing/.conda/envs/dreamgaussian/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "main.py", line 404, in save_model
    mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/dreamgaussian/gs_renderer.py", line 309, in extract_mesh
    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/dreamgaussian/mesh_utils.py", line 113, in clean_mesh
    threshold=pml.PercentageValue(v_pct)
AttributeError: module 'pymeshlab' has no attribute 'PercentageValue'
```
we do not seem to be the only one: https://github.com/dreamgaussian/dreamgaussian/issues/35


Why would you even define requirements if your code does not comply...? Jeez.



We can fix this by running

```bash
pip install pymeshlab==0.2
```

after activivating the env but before running the training loop. However, now we get:


```error
AttributeError: 'pymeshlab.pmeshlab.MeshSet' object has no attribute 'meshing_remove_unreferenced_vertices'
```

running "conda update pymeshlab" instead does not work as it will complain the conda env does not have that package installed, prob bc we installed everything using pip??


running "conda install pymeshlabl" will give:

```error
PackagesNotFoundError: The following packages are not available from current channels:

  - pymeshlab

Current channels:

  - https://repo.anaconda.com/pkgs/main/linux-64
  - https://repo.anaconda.com/pkgs/main/noarch
  - https://repo.anaconda.com/pkgs/r/linux-64
  - https://repo.anaconda.com/pkgs/r/noarch

```

running "pip install pymeshlab==2022.02" will give error again:

```error
AttributeError: module 'pymeshlab' has no attribute 'PercentageValue'
```

we can fix all pyhmeshlab related errors by doing a clean install of the environment and monkey patching the the mesh_utils.py file as follows, so it uses other method names: https://github.com/3DTopia/LGM/issues/2


>! NOTE: probably everything would also be fixed by using pymeshlab 2023.12, as defined in the dreamgaussian requirements.lock.txt. However, it seems we cannot easily install that version of Pymeshlab on our particular partition (genoa) of Snellius... But also not sure what pymeshlab version we are actually using now... ow do we check that?#TODO. 

