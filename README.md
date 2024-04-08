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

## echo errors:
```bash
awk 'tolower($0) ~ /error/ {print; err=1; next} /^[ \t]/ && err {print; next} {err=0}' filename

```

## Refernces:
- https://servicedesk.surf.nl/wiki/display/WIKI/Software+policy+Snellius#SoftwarepolicySnellius-UseofAnacondaandMinicondaenvironmentsonSnellius

# Working with Git Submodules
- https://github.blog/2016-02-01-working-with-submodules/