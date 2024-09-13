# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import earthkit.regrid as ekr
from earthkit.data.indexing.fieldlist import FieldArray

from .base import RequestBasedInput

LOG = logging.getLogger(__name__)


def _noop(x):
    return x


class NewDataField:
    def __init__(self, field, data, param):
        self.field = field
        self.data = data
        self.param = param

    def to_numpy(self, flatten=False, dtype=None, index=None):
        data = self.data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        return data

    def metadata(self, key, *args, **kwargs):
        if key == "param":
            return self.param
        return self.field.metadata(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.field, name)

    def __repr__(self) -> str:
        return repr(self.field)


class Interpolate:
    def __init__(self, grid, source):
        self.grid = list(grid) if isinstance(grid, tuple) else grid
        self.source = list(source) if isinstance(source, tuple) else source

    def __call__(self, ds):
        result = []
        for f in ds:
            data = ekr.interpolate(f.to_numpy(), dict(grid=self.source), dict(grid=self.grid))
            result.append(NewDataField(f, data))
        return FieldArray(result)


def make_z_from_gh(previous):
    g = 9.80665  # Same a pgen

    def _proc(ds):

        ds = previous(ds)

        result = []
        for f in ds:
            if f.metadata("param") == "gh":
                result.append(NewDataField(f, f.to_numpy() * g, param="z"))
            else:
                result.append(f)
        return FieldArray(result)

    return _proc


class OpenDataInput(RequestBasedInput):
    WHERE = "OPENDATA"

    RESOLS = {
        (0.25, 0.25): ("0p25", (0.25, 0.25), False),
        "N320": ("0p25", (0.25, 0.25), True),
        "O96": ("0p25", (0.25, 0.25), True),
    }

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def _adjust(self, kwargs):
        if "level" in kwargs:
            # OpenData uses levelist instead of level
            kwargs["levelist"] = kwargs.pop("level")

        grid = kwargs.pop("grid")
        if isinstance(grid, list):
            grid = tuple(grid)

        kwargs["resol"], source, interp = self.RESOLS[grid]
        r = dict(**kwargs)
        r.update(self.owner.retrieve)

        if interp:
            return Interpolate(grid, source)
        else:
            return _noop

    def pl_load_source(self, **kwargs):
        pproc = self._adjust(kwargs)
        kwargs["levtype"] = "pl"

        param = [p.lower() for p in kwargs["param"]]
        assert isinstance(param, (list, tuple))

        if "z" in param:
            param = list(param)
            param.remove("z")
            if "gh" not in param:
                param.append("gh")
            kwargs["param"] = param
            pproc = make_z_from_gh(pproc)

        logging.debug("load source ecmwf-open-data %s", kwargs)
        return pproc(ekd.from_source("ecmwf-open-data", **kwargs))

    def sfc_load_source(self, **kwargs):
        pproc = self._adjust(kwargs)
        kwargs["levtype"] = "sfc"
        logging.debug("load source ecmwf-open-data %s", kwargs)
        return pproc(ekd.from_source("ecmwf-open-data", **kwargs))

    def ml_load_source(self, **kwargs):
        pproc = self._adjust(kwargs)
        kwargs["levtype"] = "ml"
        logging.debug("load source ecmwf-open-data %s", kwargs)
        return pproc(ekd.from_source("ecmwf-open-data", **kwargs))
