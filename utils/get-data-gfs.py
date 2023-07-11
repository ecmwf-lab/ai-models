#!/usr/bin/env python3

# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import sys

import climetlab as cml
import tqdm

date = sys.argv[1]
time = sys.argv[2]
output = sys.argv[3]

time = int(time)
if time > 100:
    time /= 100
time = int(time)
time = f"{time:02d}"

gfs = cml.load_source(
    "url-pattern",
    "https://data.rda.ucar.edu/ds084.1/{year}/{date}/"
    "gfs.0p25.{date}{time}.f{step}.grib2",
    date=date,
    time=time,
    step="000",
    year=date[:4],
)


# gfs.save("gfs.grib")

param_sfc = ["prmsl", "10u", "10v", "2t"]
param_level_pl = (
    ["gh", "q", "t", "u", "v"],
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)

param, level = param_level_pl
print("Sel+order", param, level)
fields_pl = gfs.sel(
    param=param,
    level=level,
    levtype="pl",
).order_by(
    param=param,
    level=level,
)

print("Sel+order", param_sfc)

fields_sfc = gfs.sel(
    param=param_sfc,
    levtype="sfc",
).order_by(param=param_sfc)

print("Write", output)
out = cml.new_grib_output(output)

G = {"gh": 9.80665}
PARAM = {"gh": "z", "prmsl": "msl"}

for f in tqdm.tqdm(fields_pl + fields_sfc):
    param = f.metadata("shortName")
    out.write(
        f.to_numpy() * G.get(param, 1),
        template=f,
        centre=98,
        setLocalDefinition=1,
        subCentre=7,
        localDefinitionNumber=1,
        param=PARAM.get(param, param),
    )
