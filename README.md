# ai-models

The `ai-models` command is used to run AI-based weather forecasting models. These models needs to be installed independently.

Currently, too models can be installed:

```bash
pip install ai-models-panguweather
pip install ai-models-fourcastnet
```

See [ai-models-panguweather](https://github.com/ecmwf-lab/ai-models-panguweather) and [ai-models-fourcastnet](https://github.com/ecmwf-lab/ai-models-fourcastnet).
 for more details about these models.

## Running the models

## Weights (a.k.a assets)



## Input data

## Performances

Although the models will run on a CPU, they will run very slowly.

:warning: **We strongly recommend to run these models on a computer equipped with  a GPU.**

## Command line options

It has the following options:

- `--help`: Displays this help message.
- `--models`: Lists all available models.
- `--debug`: Turns on debug mode. This will print additional information to the console.

### Input

- `--input INPUT`: The input source for the model. This can be a file, a directory, or a URL.
- `--file FILE`: The specific file to use as input. This option is only relevant if the input source is a directory.

- `--date DATE`: The analysis date for the model. This defaults to yesterday.
- `--time TIME`: The analysis time for the model. This defaults to 12:00 PM.

### Output

- `--output OUTPUT`: The output destination for the model. This can be a file, a directory, or a URL.
- `--path PATH`: The path to write the output of the model. This defaults to the current working directory.


### Assets management
- `--assets ASSETS`: The path to the directory containing the model assets. These assets are typically weights and other files that are needed to run the model.
- `--assets-sub-directory`: The subdirectory of the assets directory to load assets from. This is only relevant if the assets directory contains multiple subdirectories.
- `--download-assets`: Downloads the assets if they do not exist.

### Misc. options
- `--expver EXPVER`: The experiment version of the model output. This is typically used to track different versions of the model.
