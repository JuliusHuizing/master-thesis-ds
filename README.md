# Master Thesis

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
>

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