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