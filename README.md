# Overview



# Job Scripts and Associated Configuration Files

This project includes several job scripts, each associated with one or more configuration files. Below is a detailed explanation for each job script, including the associated configuration files, their purpose, and additional notes.

## run_pipeline.job
Transforms a single image into a 3D Gaussian splatting scene (.ply file) and optionally into a mesh (.obj) file
if the second stage of the pipeline is actived in the configuration file. In addtion, after each stage, the pipeline computes valuation metrics (clip scores for quantivative evaluation; images for qualitative evaluation) and stores them in 
the .csv files / output directories specified in the configuration file.

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `batch_config.yaml`         | Configuration for batch processing.          | Settings for executing batch processing tasks. Includes parameters such as input file paths, output directories, and batch size. |
| `config.yaml`               | General configuration file for the project.  | Centralized configuration settings used across different scripts and modules. Includes paths, API keys, and other essential settings. |

> [!NOTE]
> This job uses the DreamGaussianV2 environment. If you haven't done so already, install it before running the job:
> ```bash
> sbatch jobs/install_dream_gaussian_environment.job
>```

### Running the job:

```bash
sbatch run_pipline.job
```


### run_compare.job

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `compare_config.woreg.yaml` | Configuration for comparison operations without regular expressions. | Holds settings for comparing datasets or results without using regex. Focuses on direct comparison parameters. |
| `compare_config.yaml`       | Configuration for comparison operations.     | Parameters for comparing datasets or model results. Includes regex settings and other comparison-specific options. |

### run_gridsearch.job

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `grid_search.yaml`          | Configuration for grid search operations.    | Defines the parameter grid for hyperparameter tuning. Contains settings like parameter ranges, scoring metrics, and cross-validation details. |



### run_sugar.job

| Configuration File          | Purpose                                      | Note                                                                 |
|-----------------------------|----------------------------------------------|----------------------------------------------------------------------|
| `batch_config.woreg.yaml`   | Configuration for batch processing without regular expressions. | Contains settings related to batch processing tasks without using regex. Includes input file paths, output directories, and batch size. |
