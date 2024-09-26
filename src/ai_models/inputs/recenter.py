# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import numpy as np
import tqdm
from earthkit.data.core.temporary import temp_file

LOG = logging.getLogger(__name__)

CHECKED = set()


def _init_recenter(ds, f):

    # For now, we only support the 0.25x0.25 grid from OPENDATA (centered on the greenwich meridian)

    latitudeOfFirstGridPointInDegrees = f.metadata("latitudeOfFirstGridPointInDegrees")
    longitudeOfFirstGridPointInDegrees = f.metadata("longitudeOfFirstGridPointInDegrees")
    latitudeOfLastGridPointInDegrees = f.metadata("latitudeOfLastGridPointInDegrees")
    longitudeOfLastGridPointInDegrees = f.metadata("longitudeOfLastGridPointInDegrees")
    iDirectionIncrementInDegrees = f.metadata("iDirectionIncrementInDegrees")
    jDirectionIncrementInDegrees = f.metadata("jDirectionIncrementInDegrees")
    scanningMode = f.metadata("scanningMode")
    Ni = f.metadata("Ni")
    Nj = f.metadata("Nj")

    assert scanningMode == 0
    assert latitudeOfFirstGridPointInDegrees == 90
    assert longitudeOfFirstGridPointInDegrees == 180
    assert latitudeOfLastGridPointInDegrees == -90
    assert longitudeOfLastGridPointInDegrees == 179.75
    assert iDirectionIncrementInDegrees == 0.25
    assert jDirectionIncrementInDegrees == 0.25

    assert Ni == 1440
    assert Nj == 721

    shape = (Nj, Ni)
    roll = -Ni // 2
    axis = 1

    key = (
        latitudeOfFirstGridPointInDegrees,
        longitudeOfFirstGridPointInDegrees,
        latitudeOfLastGridPointInDegrees,
        longitudeOfLastGridPointInDegrees,
        iDirectionIncrementInDegrees,
        jDirectionIncrementInDegrees,
        Ni,
        Nj,
    )

    ############################

    if key not in CHECKED:
        lon = ekd.from_source("forcings", ds, param=["longitude"], date=f.metadata("date"))[0]
        assert np.all(np.roll(lon.to_numpy(), roll, axis=axis)[:, 0] == 0)
        CHECKED.add(key)

    return (shape, roll, axis, dict(longitudeOfFirstGridPointInDegrees=0, longitudeOfLastGridPointInDegrees=359.75))


def recenter(ds):

    tmp = temp_file()

    out = ekd.new_grib_output(tmp.path)

    for f in tqdm.tqdm(ds, delay=0.5, desc="Recentering", leave=False):

        shape, roll, axis, metadata = _init_recenter(ds, f)

        data = f.to_numpy()
        assert data.shape == shape, (data.shape, shape)

        data = np.roll(data, roll, axis=axis)

        out.write(data, template=f, **metadata)

    out.close()

    result = ekd.from_source("file", tmp.path)
    result._tmp = tmp

    return result
