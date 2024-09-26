# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import tqdm
from earthkit.data.core.temporary import temp_file
from earthkit.data.indexing.fieldlist import FieldArray

LOG = logging.getLogger(__name__)

G = 9.80665  # Same a pgen


def make_z_from_gh(ds):

    tmp = temp_file()

    out = ekd.new_grib_output(tmp.path)
    other = []

    for f in tqdm.tqdm(ds, delay=0.5, desc="GH to Z", leave=False):

        if f.metadata("param") == "gh":
            out.write(f.to_numpy() * G, template=f, param="z")
        else:
            other.append(f)

    out.close()

    result = FieldArray(other) + ekd.from_source("file", tmp.path)
    result._tmp = tmp

    return result
