# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sys
import time
from functools import cached_property

import climetlab as cml
import entrypoints
from climetlab.utils.humanize import seconds
from multiurl import download

LOG = logging.getLogger(__name__)


class MarsInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        LOG.info("Loading surface fields from MARS")
        request = dict(
            date=self.owner.date,
            time=self.owner.time,
            param=self.owner.param_sfc,
            grid=self.owner.grid,
            area=self.owner.area,
            levtype="sfc",
        )
        return cml.load_source("mars", request)

    @cached_property
    def fields_pl(self):
        LOG.info("Loading pressure fields from MARS")
        param, level = self.owner.param_level_pl
        request = dict(
            date=self.owner.date,
            time=self.owner.time,
            param=param,
            level=level,
            grid=self.owner.grid,
            area=self.owner.area,
            levtype="pl",
        )
        return cml.load_source("mars", request)

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl


class CdsInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        LOG.info("Loading surface fields from the CDS")
        request = dict(
            product_type="reanalysis",
            date=self.owner.date,
            time=self.owner.time,
            param=self.owner.param_sfc,
            grid=self.owner.grid,
            area=self.owner.area,
            levtype="sfc",
        )
        return cml.load_source("cds", "reanalysis-era5-single-levels", request)

    @cached_property
    def fields_pl(self):
        LOG.info("Loading pressure fields  from the CDS")
        param, level = self.owner.param_level_pl
        request = dict(
            product_type="reanalysis",
            date=self.owner.date,
            time=self.owner.time,
            param=param,
            level=level,
            grid=self.owner.grid,
            area=self.owner.area,
            levtype="pl",
        )
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", request)

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl


class FileInput:
    def __init__(self, owner, file, **kwargs):
        self.file = file
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        return cml.load_source("file", self.file).sel(levtype="sfc")

    @cached_property
    def fields_pl(self):
        return cml.load_source("file", self.file).sel(levtype="pl")

    @cached_property
    def all_fields(self):
        return cml.load_source("file", self.file)


class FileOutput:
    def __init__(self, owner, path, expver, **kwargs):
        if path is None:
            path = kwargs["model"] + ".grib"

        if expver is None:
            expver = owner.expver

        LOG.info("Writting results to %s", path)
        self.path = path
        self.owner = owner
        self.output = cml.new_grib_output(
            path,
            split_output=True,
            class_="ml",
            expver=expver,
            edition=2,
        )

    def write(self, *args, **kwargs):
        self.output.write(*args, **kwargs)


INPUTS = dict(
    mars=MarsInput,
    file=FileInput,
    cds=CdsInput,
)

OUTPUTS = dict(file=FileOutput)


class Timer:
    def __init__(self, title):
        self.title = title
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("%s: %s.", self.title, seconds(elapsed))


class Stepper:
    def __init__(self, step, lead_time):
        self.step = step
        self.lead_time = lead_time
        self.start = time.time()
        self.last = self.start
        self.num_steps = lead_time // step
        LOG.info("Starting inference for %s steps (%sh).", self.num_steps, lead_time)

    def __enter__(self):
        return self

    def __call__(self, i, step):
        now = time.time()
        elapsed = now - self.start
        speed = (i + 1) / elapsed
        eta = (self.num_steps - i) / speed
        LOG.info(
            "Done %s out of %s in %s (%sh), ETA: %s.",
            i + 1,
            self.num_steps,
            seconds(now - self.last),
            step,
            seconds(eta),
        )
        self.last = now

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("Elapsed: %s.", seconds(elapsed))
        LOG.info("Average: %s per step.", seconds(elapsed / self.num_steps))


class Model:
    assets_extra_dir = None

    def __init__(self, input, output, download_assets, **kwargs):
        self.input = INPUTS[input](self, **kwargs)
        self.output = OUTPUTS[output](self, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.assets_sub_directory:
            if self.assets_extra_dir is not None:
                self.assets += self.assets_extra_dir

        LOG.debug("Asset directory is %s", self.assets)

        try:
            # For CliMetLab, when date=-1
            self.date = int(self.date)
        except ValueError:
            pass

        if download_assets:
            self.download_assets(**kwargs)

    @cached_property
    def fields_pl(self):
        return self.input.fields_pl

    @cached_property
    def fields_sfc(self):
        return self.input.fields_sfc

    @cached_property
    def all_fields(self):
        return self.input.all_fields

    def write(self, *args, **kwargs):
        self.output.write(*args, **kwargs)

    def download_assets(self, **kwargs):
        for file in self.download_files:
            asset = os.path.realpath(os.path.join(self.assets, file))
            if not os.path.exists(asset):
                LOG.info("Downloading %s", asset)
                download(self.download_url.format(file=file), asset + ".download")
                os.rename(asset + ".download", asset)

    @cached_property
    def device(self):
        import torch

        device = "cpu"

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"

        if torch.cuda.is_available() and torch.cuda.is_built():
            device = "cuda"

        LOG.info(
            "Using device '%s'. The speed of inference depends greatly on the device.",
            device.upper(),
        )

        return device

    @cached_property
    def providers(self):
        import platform

        import GPUtil
        import onnxruntime as ort

        providers = []

        if GPUtil.getAvailable():
            providers += [
                "CUDAExecutionProvider",  # CUDA
            ]

        if sys.platform == "darwin":
            if platform.machine() == "arm64":
                # This one is not working with error: CoreML does not support input dim > 16384
                # providers += ["CoreMLExecutionProvider"]
                pass

        providers += [
            "CPUExecutionProvider",  # CPU
        ]

        LOG.info(
            "Using device '%s'. The speed of inference depends greatly on the device.",
            ort.get_device(),
        )
        return providers

    def timer(self, title):
        return Timer(title)

    def stepper(self, step):
        return Stepper(step, self.lead_time)


def available_models():
    result = {}
    for e in entrypoints.get_group_all("ai_models.model"):
        result[e.name] = e
    return result


def load_model(name, **kwargs):
    return available_models()[name].load()(**kwargs)
