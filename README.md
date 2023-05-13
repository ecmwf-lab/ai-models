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

To run model, make sure it has been installed, then simply run:

```bash
ai-models <model-name>
```

By default, the model will be run for 10 days lead time, getting its initial conditions from ECMWF's MARS archive. See below how these defaults can be changed.

## Assets

AI models rely on weigths that have been created during training. So the first time you run a model, you will need to download the trained weigths, as well possibly some other assets.

The following command will download the assets needed by the model before running it. Assets will only be downloaded if needed, and they will be written  in the current directory,

```bash
ai-models --download-assets <model-name>
```

You can provide a different directory to store the assets:

```bash
ai-models --assets <some-directory> --download-assets <model-name>
```

Then later, simply use:

```bash
ai-models --assets <some-directory>  <model-name>
```

or

```bash
export AI_MODELS_ASSETS=<some-directory>
ai-models <model-name>
```

To keep the assets directory organised, you can use the `--assets-sub-directory` option that will store the assets of each models in its own subdirectory of the `--assets` directory.

## Input data

### From MARS

### From the Copernicus Climate Data Store (CDS)

### From a local file

## Performances

Although the models will run on a CPU, they will run very slowly. A 10-day forecast will take around one minute on a modern GPU, while the same forecast can take several hours on a CPU.

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
