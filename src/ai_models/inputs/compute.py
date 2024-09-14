# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from earthkit.data.indexing.fieldlist import FieldArray

from .transform import NewDataField
from .transform import NewMetadataField

LOG = logging.getLogger(__name__)


def make_z_from_gh(previous):
    g = 9.80665  # Same a pgen

    def _proc(ds):

        ds = previous(ds)

        result = []
        for f in ds:
            if f.metadata("param") == "gh":
                result.append(NewMetadataField(NewDataField(f, f.to_numpy() * g), param="z"))
            else:
                result.append(f)
        return FieldArray(result)

    return _proc
