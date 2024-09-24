# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


class WrappedField:
    def __init__(self, field):
        self._field = field

    def __getattr__(self, name):
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)


class NewDataField(WrappedField):
    def __init__(self, field, data):
        super().__init__(field)
        self._data = data
        self.shape = data.shape

    def to_numpy(self, flatten=False, dtype=None, index=None):
        data = self._data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        if index is not None:
            data = data[index]
        return data


class NewMetadataField(WrappedField):
    def __init__(self, field, **kwargs):
        super().__init__(field)
        self._metadata = kwargs

    def metadata(self, *args, **kwargs):
        if len(args) == 1 and args[0] in self._metadata:
            return self._metadata[args[0]]
        return self._field.metadata(*args, **kwargs)
