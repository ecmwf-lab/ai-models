# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import lru_cache

import earthkit.data as ekd
import earthkit.regrid as ekr
import tqdm
from earthkit.data.core.temporary import temp_file

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def ll_0p25():
    return dict(
        latitudeOfFirstGridPointInDegrees=90,
        longitudeOfFirstGridPointInDegrees=0,
        latitudeOfLastGridPointInDegrees=-90,
        longitudeOfLastGridPointInDegrees=359.75,
        iDirectionIncrementInDegrees=0.25,
        jDirectionIncrementInDegrees=0.25,
        Ni=1440,
        Nj=721,
        gridType="regular_ll",
    )


@lru_cache(maxsize=None)
def gg_n320():
    import eccodes

    sample = None
    result = {}
    try:
        sample = eccodes.codes_new_from_samples("reduced_gg_pl_320_grib2", eccodes.CODES_PRODUCT_GRIB)

        for key in ("N", "Ni", "Nj"):
            result[key] = eccodes.codes_get(sample, key)

        for key in (
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "latitudeOfLastGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
        ):
            result[key] = eccodes.codes_get_double(sample, key)

        pl = eccodes.codes_get_long_array(sample, "pl")
        result["pl"] = pl.tolist()
        result["gridType"] = "reduced_gg"

        return result

    finally:
        if sample is not None:
            eccodes.codes_release(sample)


METADATA = {"N320": gg_n320, (0.25, 0.25): ll_0p25}


class Interpolate:
    def __init__(self, *, source, target):
        self.target = tuple(target) if isinstance(target, list) else target
        self.source = list(source) if isinstance(source, list) else source
        self.metadata = METADATA[self.target]()

    def __repr__(self):
        return f"Interpolate(source={self.source}, target={self.target})"

    def __call__(self, ds):
        tmp = temp_file()

        out = ekd.new_grib_output(tmp.path)
        for f in tqdm.tqdm(ds, delay=0.5, desc="Interpolating", leave=False):
            data = ekr.interpolate(f.to_numpy(), dict(grid=self.source), dict(grid=self.target))
            out.write(data, template=f, metadata=self.metadata)

        out.close()

        result = ekd.from_source("file", tmp.path)
        result._tmp = tmp

        return result

    def interpolate(self, values):
        data = ekr.interpolate(values, dict(grid=self.source), dict(grid=self.target))
        return data, self.metadata
