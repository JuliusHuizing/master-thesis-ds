# Master Thesis

# Snellius 
## Connecting to Snellius:

```bash
ssh -X jhuizing@snellius.surf.nl
```

## Set-up
Ensure you clone the repository recursively such that all submodules get loadeded correctly:

```bash
git clone --recursive https://github.com/JuliusHuizing/master-thesis-ds
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


## Working with Git Submodules
- https://github.blog/2016-02-01-working-with-submodules/