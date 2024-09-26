# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import os

import earthkit.data as ekd
from earthkit.data.core.temporary import temp_file
from earthkit.data.indexing.fieldlist import FieldArray
from multiurl import download

from .base import RequestBasedInput
from .compute import make_z_from_gh
from .interpolate import Interpolate
from .recenter import recenter
from .transform import NewMetadataField

LOG = logging.getLogger(__name__)

CONSTANTS = (
    "z",
    "sdor",
    "slor",
)

CONSTANTS_URL = "https://get.ecmwf.int/repository/test-data/ai-models/opendata/constants-{resol}.grib2"

RESOLS = {
    (0.25, 0.25): ("0p25", (0.25, 0.25), False, False, {}),
    (0.1, 0.1): (
        "0p25",
        (0.25, 0.25),
        True,
        True,
        dict(
            longitudeOfLastGridPointInDegrees=359.9,
            iDirectionIncrementInDegrees=0.1,
            jDirectionIncrementInDegrees=0.1,
            Ni=3600,
            Nj=1801,
        ),
    ),
    # "N320": ("0p25", (0.25, 0.25), True, False, dict(gridType='reduced_gg')),
    # "O96": ("0p25", (0.25, 0.25), True, False, dict(gridType='reduced_gg', )),
}


def _identity(x):
    return x


class OpenDataInput(RequestBasedInput):
    WHERE = "OPENDATA"

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def _adjust(self, kwargs):

        kwargs.setdefault("step", 0)

        if "level" in kwargs:
            # OpenData uses levelist instead of level
            kwargs["levelist"] = kwargs.pop("level")

        if "area" in kwargs:
            kwargs.pop("area")

        grid = kwargs.pop("grid")
        if isinstance(grid, list):
            grid = tuple(grid)

        kwargs["resol"], source, interp, oversampling, metadata = RESOLS[grid]
        r = dict(**kwargs)
        r.update(self.owner.retrieve)

        if interp:

            logging.info("Interpolating input data from %s to %s.", source, grid)
            if oversampling:
                logging.warning("This will oversample the input data.")
            return Interpolate(grid, source, metadata)
        else:
            return _identity

    def pl_load_source(self, **kwargs):

        gh_to_z = _identity
        interpolate = self._adjust(kwargs)

        kwargs["levtype"] = "pl"
        request = kwargs.copy()

        param = [p.lower() for p in kwargs["param"]]
        assert isinstance(param, (list, tuple))

        if "z" in param:
            logging.warning("Parameter 'z' on pressure levels is not available in ECMWF open data, using 'gh' instead")
            param = list(param)
            param.remove("z")
            if "gh" not in param:
                param.append("gh")
            kwargs["param"] = param
            gh_to_z = make_z_from_gh

        logging.info("ecmwf-open-data %s", kwargs)

        opendata = recenter(ekd.from_source("ecmwf-open-data", **kwargs))
        opendata = gh_to_z(opendata)
        opendata = interpolate(opendata)

        return self.check_pl(opendata, request)

    def constants(self, constant_params, request, kwargs):
        if len(constant_params) == 1:
            logging.warning(
                f"Single level parameter '{constant_params[0]}' is"
                " not available in ECMWF open data, using constants.grib2 instead"
            )
        else:
            logging.warning(
                f"Single level parameters {constant_params} are"
                " not available in ECMWF open data, using constants.grib2 instead"
            )

        cachedir = os.path.expanduser("~/.cache/ai-models")
        constants_url = CONSTANTS_URL.format(resol=request["resol"])
        basename = os.path.basename(constants_url)

        if not os.path.exists(cachedir):
            os.makedirs(cachedir)

        path = os.path.join(cachedir, basename)

        if not os.path.exists(path):
            logging.info("Downloading %s to %s", constants_url, path)
            download(constants_url, path + ".tmp")
            os.rename(path + ".tmp", path)

        ds = ekd.from_source("file", path)
        ds = ds.sel(param=constant_params)

        tmp = temp_file()

        out = ekd.new_grib_output(tmp.path)

        for f in ds:
            out.write(
                f.to_numpy(),
                template=f,
                date=kwargs["date"],
                time=kwargs["time"],
                step=kwargs.get("step", 0),
            )

        out.close()

        result = ekd.from_source("file", tmp.path)
        result._tmp = tmp

        return result

    def sfc_load_source(self, **kwargs):
        interpolate = self._adjust(kwargs)

        kwargs["levtype"] = "sfc"
        request = kwargs.copy()

        param = [p.lower() for p in kwargs["param"]]
        assert isinstance(param, (list, tuple))

        constant_params = []
        param = list(param)
        for c in CONSTANTS:
            if c in param:
                param.remove(c)
                constant_params.append(c)

        if constant_params:
            constants = self.constants(constant_params, request, kwargs)
        else:
            constants = ekd.from_source("empty")

        kwargs["param"] = param

        opendata = recenter(ekd.from_source("ecmwf-open-data", **kwargs))
        opendata = opendata + constants
        opendata = interpolate(opendata)

        # Fix grib2/eccodes bug

        opendata = FieldArray([NewMetadataField(f, levelist=None) for f in opendata])

        return self.check_sfc(opendata, request)

    def ml_load_source(self, **kwargs):
        interpolate = self._adjust(kwargs)
        kwargs["levtype"] = "ml"
        request = kwargs.copy()

        opendata = recenter(ekd.from_source("ecmwf-open-data", **kwargs))
        opendata = interpolate(opendata)

        return self.check_ml(opendata, request)

    def check_pl(self, ds, request):
        self._check(ds, "PL", request, "param", "levelist")
        return ds

    def check_sfc(self, ds, request):
        self._check(ds, "SFC", request, "param")
        return ds

    def check_ml(self, ds, request):
        self._check(ds, "ML", request, "param", "levelist")
        return ds

    def _check(self, ds, what, request, *keys):

        def _(p):
            if len(p) == 1:
                return p[0]

        expected = set()
        for p in itertools.product(*[request[key] for key in keys]):
            expected.add(p)

        found = set()
        for f in ds:
            found.add(tuple(f.metadata(key) for key in keys))

        missing = expected - found
        if missing:
            missing = [_(p) for p in missing]
            if len(missing) == 1:
                raise ValueError(f"The following {what} parameter '{missing[0]}' is not available in ECMWF open data")
            raise ValueError(f"The following {what} parameters {missing} are not available in ECMWF open data")

        extra = found - expected
        if extra:
            extra = [_(p) for p in extra]
            if len(extra) == 1:
                raise ValueError(f"Unexpected {what} parameter '{extra[0]}' from ECMWF open data")
            raise ValueError(f"Unexpected {what} parameters {extra} from ECMWF open data")
