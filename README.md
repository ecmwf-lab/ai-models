# ai-models

The `ai-models` command is used to run AI-based weather forecasting models. These models need to be installed independently.

## Usage

Although the source code `ai-models` and its plugins are available under open sources licences, some model weights may be available under a different licence. For example some models make their weights available under the CC-BY-NC-SA 4.0 license, which does not allow commercial use. For more informations, please check the license associated with each model on their main home page, that we link from each of the corresponding plugins.

## Prerequisites

Before using the `ai-models` command, ensure you have the following prerequisites:

- Python 3.10 (it may work with different versions, but it has been tested with 3.10 on Linux/MacOS).
- An ECMWF and/or CDS account for accessing input data (see below for more details).
- A computed with a GPU for optimal performance (strongly recommended).

## Installation

To install the `ai-models` command, run the following command:

```bash
pip install ai-models
```

## Available Models

Currently, four models can be installed:

```bash
pip install ai-models-panguweather
pip install ai-models-fourcastnet
pip install ai-models-graphcast  # Install details at https://github.com/ecmwf-lab/ai-models-graphcast
pip install ai-models-fourcastnetv2
```

See [ai-models-panguweather](https://github.com/ecmwf-lab/ai-models-panguweather), [ai-models-fourcastnet](https://github.com/ecmwf-lab/ai-models-fourcastnet),
 [ai-models-fourcastnetv2](https://github.com/ecmwf-lab/ai-models-fourcastnetv2) and [ai-models-graphcast](https://github.com/ecmwf-lab/ai-models-graphcast) for more details about these models.

## Running the models

To run model, make sure it has been installed, then simply run:

```bash
ai-models <model-name>
```

Replace `<model-name>` with the name of the specific AI model you want to run.

By default, the model will be run for a 10-day lead time (240 hours), using yesterday's 12Z analysis from ECMWF's MARS archive.

To produce a 15 days forecast, use the `--lead-time HOURS` option:

```bash
ai-models --lead-time 360 <model-name>
```

You can change the other defaults using the available command line options, as described below.

## Performances Considerations

The AI models can run on a CPU; however, they perform significantly better on a GPU. A 10-day forecast can take several hours on a CPU but only around one minute on a modern GPU.

:warning: **We strongly recommend running these models on a computer equipped with a GPU for optimal performance.**

It you see the following message when running a model, it means that the ONNX runtime was not able to find a the CUDA libraries on your system:
> [W:onnxruntime:Default, onnxruntime_pybind_state.cc:541 CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Please reference <https://onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements> to ensure all dependencies are met.

To fix this issue, we suggest that you install `ai-models` in a [conda](https://docs.conda.io/en/latest/) environment and install the CUDA libraries in that environment. For example:

```bash
conda create -n ai-models python=3.10
conda activate ai-models
conda install cudatoolkit
pip install ai-models
...
```

## Assets

The AI models rely on weights and other assets created during training. The first time you run a model, you will need to download the trained weights and any additional required assets.

To download the assets before running a model, use the following command:

```bash
ai-models --download-assets <model-name>
```

The assets will be downloaded if needed and stored in the current directory. You can provide a different directory to store the assets:

```bash
ai-models --download-assets --assets <some-directory> <model-name>
```

Then, later on, simply use:

```bash
ai-models --assets <some-directory>  <model-name>
```

or

```bash
export AI_MODELS_ASSETS=<some-directory>
ai-models <model-name>
```

For better organisation of the assets directory, you can use the `--assets-sub-directory` option. This option will store the assets of each model in its own subdirectory within the specified assets directory.

## Input data

The models require input data (initial conditions) to run. You can provide the input data using different sources, as described below:

### From MARS

By default, `ai-models`  use yesterday's 12Z analysis from ECMWF, fetched from the Centre's MARS archive using the [ECMWF WebAPI](https://www.ecmwf.int/en/computing/software/ecmwf-web-api). You will need an ECMWF account to access that service.

To change the date or time, use the `--date` and `--time` options, respectively:

```bash
ai-models --date YYYYMMDD --time HHMM <model-name>
```

### From the CDS

You can start the models using ERA5 (ECMWF Reanalysis version 5) data for the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/). You will need to create an account on the CDS. The data will be downloaded using the [CDS API](https://cds.climate.copernicus.eu/api-how-to).

To access the CDS, simply add `--input cds` on the command line. Please note that ERA5 data is added to the CDS with a delay, so you will also have to provide a date with `--date YYYYMMDD`.

```bash
ai-models --input cds --date 20230110 --time 0000 <model-name>
```

### From a GRIB file

If you have input data in the GRIB format, you can provide the file using the `--file` option:

```bash
ai-models --file <some-grib-file> <model-name>
```

The GRIB file can contain more fields than the ones required by the model. The `ai-models` command will automatically select the necessary fields from the file.

To find out the list of fields needed by a specific model as initial conditions, use the following command:

```bash
 ai-models --fields <model-name>
 ```

## Output

By default, the model output will be written in GRIB format in a file called `<model-name>.grib`. You can change the file name with the option `--path <file-name>`. If the path you specify contains placeholders between `{` and `}`, multiple files will be created based on the [eccodes](https://confluence.ecmwf.int/display/ECC) keys. For example:

```bash
 ai-models --path 'out-{step}.grib' <model-name>
 ```

This command will create a file for each forecasted time step.

If you want to disable writing the output to a file, use the `--output none` option.

## Command line options

It has the following options:

- `--help`: Displays this help message.
- `--models`: Lists all installed models.
- `--debug`: Turns on debug mode. This will print additional information to the console.

### Input

- `--input INPUT`: The input source for the model. This can be a `mars`, `cds` or `file`.
- `--file FILE`: The specific file to use as input. This option will set `--source` to `file`.

- `--date DATE`: The analysis date for the model. This defaults to yesterday.
- `--time TIME`: The analysis time for the model. This defaults to 1200.

### Output

- `--output OUTPUT`: The output destination for the model. Values are `file` or `none`.
- `--path PATH`: The path to write the output of the model.

### Run

- `--lead-time HOURS`: The number of hours to forecast. The default is 240 (10 days).

### Assets management

- `--assets ASSETS`: Specifies the path to the directory containing the model assets. The default is the current directory, but you can override it by setting the `$AI_MODELS_ASSETS` environment variable.
- `--assets-sub-directory`: Enables organising assets in `<assets-directory>/<model-name>` subdirectories.
- `--download-assets`: Downloads the assets if they do not exist.

### Misc. options

- `--fields`: Print the list of fields needed by a model as initial conditions.
- `--expver EXPVER`: The experiment version of the model output.
- `--class CLASS`: The 'class' metadata of the model output.
- `--metadata KEY=VALUE`: Additional metadata metadata in the model output
