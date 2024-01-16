# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging

import climetlab as cml
import numpy as np

LOG = logging.getLogger(__name__)


class FileOutput:
    def __init__(self, owner, path, metadata, **kwargs):
        self._first = True
        metadata.setdefault("expver", owner.expver)
        metadata.setdefault("class", "ml")

        LOG.info("Writing results to %s.", path)
        self.path = path
        self.owner = owner

        edition = metadata.pop("edition", 2)

        self.grib_keys = dict(
            edition=edition, generatingProcessIdentifier=self.owner.version
        )
        self.grib_keys.update(metadata)

        self.output = cml.new_grib_output(
            path,
            split_output=True,
            **self.grib_keys,
        )

    def write(self, data, *args, **kwargs):
        try:
            return self.output.write(data, *args, **kwargs)
        except Exception:
            if np.isnan(data).any():
                raise ValueError(
                    f"NaN values found in field. args={args} kwargs={kwargs}"
                )
            if np.isinf(data).any():
                raise ValueError(
                    f"Infinite values found in field. args={args} kwargs={kwargs}"
                )
            raise


class HindcastReLabel:
    def __init__(self, owner, output, hindcast_reference_year, **kwargs):
        self.owner = owner
        self.output = output
        self.hindcast_reference_year = int(hindcast_reference_year)

    def write(self, *args, **kwargs):
        assert "hdate" not in kwargs, kwargs
        assert "date" not in kwargs, kwargs

        date = kwargs["template"]["date"]
        hdate = kwargs["template"]["hdate"]

        if hdate is not None:
            # Input was a hindcast
            referenceDate = self.hindcast_reference_year * 10000 + date % 10000
            assert date == referenceDate, (
                date,
                referenceDate,
                hdate,
                kwargs["template"],
            )
            kwargs["referenceDate"] = referenceDate
            kwargs["hdate"] = hdate
        else:
            referenceDate = self.hindcast_reference_year * 10000 + date % 10000
            kwargs["referenceDate"] = referenceDate
            kwargs["hdate"] = date

        handle, path = self.output.write(*args, **kwargs)

        # Check that the GRIB keys are as expected
        for key, value in itertools.chain(
            self.output.grib_keys.items(), kwargs.items()
        ):
            if key in ("template",):
                continue

            # If "param" is a string, we what to compare it to the shortName
            if key == "param":
                try:
                    float(value)
                except ValueError:
                    key = "shortName"

            assert str(handle.get(key)) == str(value), (key, handle.get(key), value)

        return handle, path


class NoneOutput:
    def __init__(self, *args, **kwargs):
        LOG.info("Results will not be written.")

    def write(self, *args, **kwargs):
        pass


OUTPUTS = dict(
    file=FileOutput,
    none=NoneOutput,
)


def get_output(name, owner, *args, **kwargs):
    result = OUTPUTS[name](owner, *args, **kwargs)
    if kwargs.get("hindcast_reference_year") is not None:
        result = HindcastReLabel(owner, result, **kwargs)
    return result


def available_outputs():
    return sorted(OUTPUTS.keys())
