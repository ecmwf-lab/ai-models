# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import earthkit.regrid as ekr
import tqdm
from earthkit.data.core.temporary import temp_file

LOG = logging.getLogger(__name__)


class Interpolate:
    def __init__(self, grid, source, metadata):
        self.grid = list(grid) if isinstance(grid, tuple) else grid
        self.source = list(source) if isinstance(source, tuple) else source
        self.metadata = metadata

    def __call__(self, ds):
        tmp = temp_file()

        out = ekd.new_grib_output(tmp.path)

        result = []
        for f in tqdm.tqdm(ds, delay=0.5, desc="Interpolating", leave=False):
            data = ekr.interpolate(f.to_numpy(), dict(grid=self.source), dict(grid=self.grid))
            out.write(data, template=f, **self.metadata)

        out.close()

        result = ekd.from_source("file", tmp.path)
        result._tmp = tmp

        print("Interpolated data", tmp.path)

        return result
