# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


class NewDataField:
    def __init__(self, field, data):
        self._field = field
        self._data = data

    def to_numpy(self, flatten=False, dtype=None, index=None):
        data = self._data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        return data

    def __getattr__(self, name):
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)


class NewMetadataField:
    def __init__(self, field, **kwargs):
        self._field = field
        self._metadata = kwargs

    def __getattr__(self, name):
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)

    def metadata(self, name, **kwargs):
        if name in self._metadata:
            return self._metadata[name]
        return self._field.metadata(name, **kwargs)
