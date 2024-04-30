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



Mesh creation seems to succeed, but then loading mesh at next step gives runtime error:

```error

AttributeError: 'Namespace' object has no attribute 'ssaa'

```


This is a bug, recently introduced, in the kiui package. We can fix this by ensuring we use a version before the bug was introduced. With the conda env activated, run:

```bash
pip install kiui==0.2.2
```

and hey, we can finally compute clip scores.



# LGM installation

Following their README instructions, we get:

```error
equirement already satisfied: mpmath>=0.19 in /gpfs/home6/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages (from sympy->torch==2.2.2->xformers) (1.3.0)
Installing collected packages: triton, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, nvidia-cusolver-cu11, nvidia-cudnn-cu11, torch, xformers
  Attempting uninstall: triton
    Found existing installation: triton 2.1.0
    Uninstalling triton-2.1.0:
      Successfully uninstalled triton-2.1.0
  Attempting uninstall: torch
    Found existing installation: torch 2.1.0+cu118
    Uninstalling torch-2.1.0+cu118:
      Successfully uninstalled torch-2.1.0+cu118
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchaudio 2.1.0+cu118 requires torch==2.1.0, but you have torch 2.2.2+cu118 which is incompatible.
torchvision 0.16.0+cu118 requires torch==2.1.0, but you have torch 2.2.2+cu118 which is incompatible.
Successfully installed nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-8.7.0.84 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.19.3 nvidia-nvtx-cu11-11.8.86 torch-2.2.2+cu118 triton-2.2.0 xformers-0.0.25.post1+cu118
Cloning into 'diff-gaussian-rasterization'...
Submodule 'third_party/glm' (https://github.com/g-truc/glm.git) registered for path 'third_party/glm'
Cloning into '/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/third_party/glm'...
Submodule path 'third_party/glm': checked out '5c46b9c07008ae65cb81ab79cd677ecc1934b903'
Processing ./diff-gaussian-rasterization
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
Building wheels for collected packages: diff-gaussian-rasterization
  Building wheel for diff-gaussian-rasterization (setup.py): started
  Building wheel for diff-gaussian-rasterization (setup.py): finished with status 'error'
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [134 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.linux-x86_64-cpython-38
      creating build/lib.linux-x86_64-cpython-38/diff_gaussian_rasterization
      copying diff_gaussian_rasterization/__init__.py -> build/lib.linux-x86_64-cpython-38/diff_gaussian_rasterization
      running build_ext
      /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/utils/cpp_extension.py:425: UserWarning: There are no g++ version bounds defined for CUDA version 11.8
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
      building 'diff_gaussian_rasterization._C' extension
      creating /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38
      creating /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/cuda_rasterizer
      Emitting ninja build file /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/build.ninja...
      Compiling objects...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      [1/5] /sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/rasterize_points.o.d -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/TH -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/THC -I/sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/include -I/home/jhuizing/.conda/envs/lgm/include/python3.8 -c -c /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/rasterize_points.cu -o /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/rasterize_points.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -I/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/third_party/glm/ -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 -std=c++17
      FAILED: /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/rasterize_points.o
      /sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/rasterize_points.o.d -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/TH -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/THC -I/sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/include -I/home/jhuizing/.conda/envs/lgm/include/python3.8 -c -c /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/rasterize_points.cu -o /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/rasterize_points.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -I/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/third_party/glm/ -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 -std=c++17
      In file included from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/util/TypeList.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/util/Metaprogramming.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/core/DispatchKeySet.h:4,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/core/Backend.h:5,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/core/Layout.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/ATen/core/TensorBody.h:12,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/ATen/core/Tensor.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/ATen/Tensor.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/autograd/function_hook.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/autograd/cpp_hook.h:2,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/autograd/variable.h:6,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/autograd/autograd.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/torch/autograd.h:3,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/torch/all.h:7,
                       from /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/extension.h:5,
                       from /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/rasterize_points.cu:13:
      /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/c10/util/C++17.h:16:2: error: #error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
       #error \
        ^~~~~
      [2/5] /sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/cuda_rasterizer/forward.o.d -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/TH -I/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torch/include/THC -I/sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/include -I/home/jhuizing/.conda/envs/lgm/include/python3.8 -c -c /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/cuda_rasterizer/forward.cu -o /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/build/temp.linux-x86_64-cpython-38/cuda_rasterizer/forward.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -I/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/third_party/glm/ -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 -std=c++17
      /gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h(151): warning #177-D: variable "p_proj" was declared but never referenced
      

```

We can get  sucessful install of the env by changing the order of installing dependencies:

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
module load 2022
module load CUDA/11.8.0
module load Anaconda3/2022.05

cd $HOME/master-thesis-ds/

git pull || true # do not exit if pull fails for some reason.

cd $HOME/master-thesis-ds/repos/LGM
conda env remove --name lgm
conda create -n lgm python=3.8 pip
source activate lgm


conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization


# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

pip install -U xformers --index-url https://download.pytorch.org/whl/cu118


# other requirements
pip install -r requirements.txt




```





But when running:

```bash
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=RunDreamGaussian
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=../logs/slurm_output_%A.out

module purge
module load 2022
module load CUDA/11.8.0
module load Anaconda3/2022.05

cd $HOME/master-thesis-ds/
git pull

cd $HOME/master-thesis-ds/repos/lgm

source activate lgm
python infer.py big --workspace workspace_test --resume workspace/model.safetensors --test_path data_test

```
we get:

```error
/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/gpfs/home6/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/torchvision/image.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/core/attention.py:22: UserWarning: xFormers is available (Attention)
  warnings.warn("xFormers is available (Attention)")
Traceback (most recent call last):
  File "infer.py", line 20, in <module>
    from core.models import LGM
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/core/models.py", line 11, in <module>
    from core.gs import GaussianRenderer
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/LGM/core/gs.py", line 7, in <module>
    from diff_gaussian_rasterization import (
  File "/home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/diff_gaussian_rasterization/__init__.py", line 15, in <module>
    from . import _C
ImportError: /home/jhuizing/.conda/envs/lgm/lib/python3.8/site-packages/diff_gaussian_rasterization/_C.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE
```


Sugars env can be isntalled without errors, but get this runtime error when running the instlal_sugar_job:

```error
Traceback (most recent call last):
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/SuGaR/train.py", line 3, in <module>
    from sugar_trainers.coarse_density import coarse_training_with_density_regularization
  File "/gpfs/home6/jhuizing/master-thesis-ds/repos/SuGaR/sugar_trainers/coarse_density.py", line 5, in <module>
    from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
  File "/home/jhuizing/.conda/envs/sugar2/lib/python3.9/site-packages/pytorch3d/loss/__init__.py", line 8, in <module>
    from .chamfer import chamfer_distance
  File "/home/jhuizing/.conda/envs/sugar2/lib/python3.9/site-packages/pytorch3d/loss/chamfer.py", line 11, in <module>
    from pytorch3d.ops.knn import knn_gather, knn_points
  File "/home/jhuizing/.conda/envs/sugar2/lib/python3.9/site-packages/pytorch3d/ops/__init__.py", line 7, in <module>
    from .ball_query import ball_query
  File "/home/jhuizing/.conda/envs/sugar2/lib/python3.9/site-packages/pytorch3d/ops/ball_query.py", line 10, in <module>
    from pytorch3d import _C
ImportError: /home/jhuizing/.conda/envs/sugar2/lib/python3.9/site-packages/pytorch3d/_C.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEENS6_INS5_6LayoutEEENS6_INS5_6DeviceEEENS6_IbEENS6_INS5_12MemoryFormatEEE
```

Again, we don't seem alone:

https://github.com/Anttwo/SuGaR/issues/136

but applying this pip install inside the activated env leads to yet another error:

```error 
UserWarning: The environment variable `CUB_HOME` was not found. NVIDIA CUB is required for compilation and can be downloaded from `https://github.com/NVIDIA/cub/releases`. You can unpack it to a location of your choice and set the environment variable `CUB_HOME` to the folder containing the `CMakeListst.txt` file.

```

Inspecting this issue page: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

indeeds points out that for Cuda >= 11.8 (which we use and which the environment.yyml of SuGAr defines as requirement) the CUB library needs to be available...

perhaps check:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
https://github.com/facebookresearch/pytorch3d/issues/1207




using python 3.8 leads to a failure when trying to solve the envrionment...

https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md