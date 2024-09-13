# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd

from .base import RequestBasedInput

LOG = logging.getLogger(__name__)


class CdsInput(RequestBasedInput):
    WHERE = "CDS"

    def pl_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return ekd.from_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return ekd.from_source("cds", "reanalysis-era5-single-levels", kwargs)

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")
