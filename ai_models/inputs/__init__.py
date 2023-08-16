# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import climetlab as cml

LOG = logging.getLogger(__name__)


class RequestBasedInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        LOG.info(f"Loading surface fields from {self.WHERE}")
        return cml.load_source(
            "multi",
            [
                self.sfc_load_source(
                    date=date,
                    time=time,
                    param=self.owner.param_sfc,
                    grid=self.owner.grid,
                    area=self.owner.area,
                    **self.owner.retrieve,
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def fields_pl(self):
        LOG.info(f"Loading pressure fields from {self.WHERE}")
        param, level = self.owner.param_level_pl
        return cml.load_source(
            "multi",
            [
                self.pl_load_source(
                    date=date,
                    time=time,
                    param=param,
                    level=level,
                    grid=self.owner.grid,
                    area=self.owner.area,
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl


class MarsInput(RequestBasedInput):
    WHERE = "MARS"

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)


class CdsInput(RequestBasedInput):
    WHERE = "CDS"

    def pl_load_source(self, **kwargs):
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        return cml.load_source("cds", "reanalysis-era5-single-levels", kwargs)


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
    def __init__(self, owner, path, metadata, **kwargs):
        self._first = True
        metadata.setdefault("expver", owner.expver)
        metadata.setdefault("class", "ml")

        LOG.info("Writting results to %s.", path)
        self.path = path
        self.owner = owner
        self.output = cml.new_grib_output(
            path,
            split_output=True,
            edition=2,
            **metadata,
        )

    def write(self, *args, **kwargs):
        return self.output.write(*args, **kwargs)


class NoneOutput:
    def __init__(self, *args, **kwargs):
        LOG.info("Results will not be written.")

    def write(self, *args, **kwargs):
        pass


INPUTS = dict(
    mars=MarsInput,
    file=FileInput,
    cds=CdsInput,
)


def get_input(name, *args, **kwargs):
    return INPUTS[name](*args, **kwargs)


def available_inputs():
    return sorted(INPUTS.keys())
