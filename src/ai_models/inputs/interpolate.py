# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.regrid as ekr
import tqdm
from earthkit.data.indexing.fieldlist import FieldArray

from .transform import NewDataField

LOG = logging.getLogger(__name__)


class Interpolate:
    def __init__(self, grid, source):
        self.grid = list(grid) if isinstance(grid, tuple) else grid
        self.source = list(source) if isinstance(source, tuple) else source

    def __call__(self, ds):
        result = []
        for f in tqdm.tqdm(ds, delay=0.5, desc="Interpolating", leave=False):
            data = ekr.interpolate(f.to_numpy(), dict(grid=self.source), dict(grid=self.grid))
            result.append(NewDataField(f, data))

        LOG.info("Interpolated %d fields. Input shape %s, output shape %s.", len(result), ds[0].shape, result[0].shape)
        return FieldArray(result)
