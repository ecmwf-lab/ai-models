# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import warnings
from functools import cached_property

import climetlab as cml
import entrypoints
import numpy as np

LOG = logging.getLogger(__name__)


class Output:
    def write(self, *args, **kwargs):
        pass

    def finalise(self, *args, **kwargs):
        pass


class FileOutputBase(Output):
    def __init__(self, owner, path, metadata, **kwargs):
        self._first = True
        metadata.setdefault("stream", "oper")
        metadata.setdefault("expver", owner.expver)
        metadata.setdefault("class", "ml")

        self.path = path
        self.owner = owner
        self.metadata = metadata

    @cached_property
    def grib_keys(self):
        edition = self.metadata.pop("edition", 2)

        _grib_keys = dict(
            edition=edition,
            generatingProcessIdentifier=self.owner.version,
        )
        _grib_keys.update(self.metadata)

        return _grib_keys

    @cached_property
    def output(self):
        return cml.new_grib_output(
            self.path,
            split_output=True,
            **self.grib_keys,
        )

    def write(self, data, *args, check=False, **kwargs):

        try:
            handle, path = self.output.write(data, *args, **kwargs)

        except Exception:
            if data is not None:
                if np.isnan(data).any():
                    raise ValueError(f"NaN values found in field. args={args} kwargs={kwargs}")
                if np.isinf(data).any():
                    raise ValueError(f"Infinite values found in field. args={args} kwargs={kwargs}")
            raise

        if check:
            # Check that the GRIB keys are as expected

            if kwargs.get("expver") is None:
                ignore = ("template", "expver", "class", "type", "stream")
            else:
                ignore = ("template",)

            for key, value in itertools.chain(self.grib_keys.items(), kwargs.items()):
                if key in ignore:
                    continue

                # If "param" is a string, we what to compare it to the shortName
                if key == "param":
                    try:
                        float(value)
                    except ValueError:
                        key = "shortName"

                assert str(handle.get(key)) == str(value), (key, handle.get(key), value)

        return handle, path


class FileOutput(FileOutputBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LOG.info("Writing results to %s", self.path)


class NoneOutput(Output):
    def __init__(self, *args, **kwargs):
        LOG.info("Results will not be written.")

    def write(self, *args, **kwargs):
        pass


class HindcastReLabel:
    def __init__(self, owner, output, hindcast_reference_year=None, hindcast_reference_date=None, **kwargs):
        self.owner = owner
        self.output = output
        self.hindcast_reference_year = int(hindcast_reference_year) if hindcast_reference_year else None
        self.hindcast_reference_date = int(hindcast_reference_date) if hindcast_reference_date else None
        assert self.hindcast_reference_year is not None or self.hindcast_reference_date is not None

    def write(self, *args, **kwargs):
        if "hdate" in kwargs:
            warnings.warn(f"Ignoring hdate='{kwargs['hdate']}' in HindcastReLabel", stacklevel=3)
            kwargs.pop("hdate")

        if "date" in kwargs:
            warnings.warn(f"Ignoring date='{kwargs['date']}' in HindcastReLabel", stacklevel=3)
            kwargs.pop("date")

        date = kwargs["template"]["date"]
        hdate = kwargs["template"]["hdate"]

        if hdate is not None:
            # Input was a hindcast
            referenceDate = (
                self.hindcast_reference_date
                if self.hindcast_reference_date is not None
                else self.hindcast_reference_year * 10000 + date % 10000
            )
            assert date == referenceDate, (
                date,
                referenceDate,
                hdate,
                kwargs["template"],
            )
            kwargs["referenceDate"] = referenceDate
            kwargs["hdate"] = hdate
        else:
            referenceDate = (
                self.hindcast_reference_date
                if self.hindcast_reference_date is not None
                else self.hindcast_reference_year * 10000 + date % 10000
            )
            kwargs["referenceDate"] = referenceDate
            kwargs["hdate"] = date

        kwargs.setdefault("check", True)

        return self.output.write(*args, **kwargs)


class NoLabelling:

    def __init__(self, owner, output, **kwargs):
        self.owner = owner
        self.output = output

    def write(self, *args, **kwargs):
        kwargs["deleteLocalDefinition"] = 1
        return self.output.write(*args, **kwargs)


def get_output(name, owner, *args, **kwargs):
    result = available_outputs()[name].load()(owner, *args, **kwargs)
    if kwargs.get("hindcast_reference_year") is not None or kwargs.get("hindcast_reference_date") is not None:
        result = HindcastReLabel(owner, result, **kwargs)
    if owner.expver is None:
        result = NoLabelling(owner, result, **kwargs)
    return result


def available_outputs():
    result = {}
    for e in entrypoints.get_group_all("ai_models.output"):
        result[e.name] = e
    return result
