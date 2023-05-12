# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from functools import cached_property

import climetlab as cml
import entrypoints
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
    def __init__(self, owner, path, **kwargs):
        LOG.info("Writting results to %s", path)
        self.path = path
        self.owner = owner
        self.output = cml.new_grib_output(
            path,
            split_output=True,
            class_="ml",
            expver=owner.expver,
            edition=2,
        )

    def write(self, *args, **kwargs):
        self.output.write(*args, **kwargs)


INPUTS = dict(mars=MarsInput, file=FileInput)

OUTPUTS = dict(file=FileOutput)


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


def available_models():
    result = {}
    for e in entrypoints.get_group_all("ai_models.model"):
        result[e.name] = e
    return result


def load_model(name, **kwargs):
    return available_models()[name].load()(**kwargs)
