# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict
from functools import cached_property

import entrypoints
from climetlab.utils.humanize import seconds
from multiurl import download

from .checkpoint import peek
from .inputs import get_input
from .outputs import get_output
from .stepper import Stepper

LOG = logging.getLogger(__name__)


class Timer:
    def __init__(self, title):
        self.title = title
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("%s: %s.", self.title, seconds(elapsed))


class ArchiveCollector:
    def __init__(self) -> None:
        self.expect = 0
        self.request = defaultdict(set)

    def add(self, field):
        self.expect += 1
        for k, v in field.items():
            self.request[k].add(str(v))


class Model:
    lagged = False
    assets_extra_dir = None
    retrieve = {}  # Extra parameters for retrieve
    version = 1  # To be overriden in subclasses

    def __init__(self, input, output, download_assets, **kwargs):
        self.input = get_input(input, self, **kwargs)
        self.output = get_output(output, self, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        # We need to call it to initialise the default args
        args = self.parse_model_args(self.model_args)
        if args:
            for k, v in vars(args).items():
                setattr(self, k, v)

        if self.assets_sub_directory:
            if self.assets_extra_dir is not None:
                self.assets += self.assets_extra_dir

        LOG.debug("Asset directory is %s", self.assets)

        try:
            # For CliMetLab, when date=-1
            self.date = int(self.date)
        except ValueError:
            pass

        if download_assets:
            self.download_assets(**kwargs)

        self.archiving = defaultdict(ArchiveCollector)
        self.created = time.time()

    @cached_property
    def fields_pl(self):
        return self.input.fields_pl

    @cached_property
    def fields_sfc(self):
        return self.input.fields_sfc

    @cached_property
    def all_fields(self):
        return self.input.all_fields

    def write(self, *args, **kwargs):
        self.collect_archive_requests(
            self.output.write(*args, **kwargs),
        )

    def collect_archive_requests(self, written):
        if self.archive_requests:
            handle, path = written
            if self.hindcast_reference_year:
                # The clone is necessary because the handle
                # does not return always return recently set keys
                handle = handle.clone()

            self.archiving[path].add(handle.as_mars())

    def finalise(self):
        if self.archive_requests:
            with open(self.archive_requests, "w") as f:
                for path, archive in self.archiving.items():
                    request = dict(source=f'"{path}"', expect=archive.expect)
                    request.update(archive.request)
                    request.update(self._requests_extra)
                    self._print_request("archive", request, file=f)

    def download_assets(self, **kwargs):
        for file in self.download_files:
            asset = os.path.realpath(os.path.join(self.assets, file))
            if not os.path.exists(asset):
                os.makedirs(os.path.dirname(asset), exist_ok=True)
                LOG.info("Downloading %s", asset)
                download(self.download_url.format(file=file), asset + ".download")
                os.rename(asset + ".download", asset)

    @property
    def asset_files(self, **kwargs):
        result = []
        for file in self.download_files:
            result.append(os.path.realpath(os.path.join(self.assets, file)))
        return result

    @cached_property
    def device(self):
        import torch

        device = "cpu"

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"

        if torch.cuda.is_available() and torch.backends.cuda.is_built():
            device = "cuda"

        LOG.info(
            "Using device '%s'. The speed of inference depends greatly on the device.",
            device.upper(),
        )

        if self.only_gpu:
            if device == "cpu":
                raise RuntimeError("GPU is not available")

        return device

    @cached_property
    def providers(self):
        import platform

        import GPUtil
        import onnxruntime as ort

        providers = []

        try:
            if GPUtil.getAvailable():
                providers += [
                    "CUDAExecutionProvider",  # CUDA
                ]
        except Exception:
            pass

        if sys.platform == "darwin":
            if platform.machine() == "arm64":
                # This one is not working with error: CoreML does not support input dim > 16384
                # providers += ["CoreMLExecutionProvider"]
                pass

        providers += [
            "CPUExecutionProvider",  # CPU
        ]

        LOG.info("ONNXRuntime providers: %s", providers)

        LOG.info(
            "Using device '%s'. The speed of inference depends greatly on the device.",
            ort.get_device(),
        )

        if self.only_gpu:
            assert isinstance(ort.get_device(), str)
            if ort.get_device() == "CPU":
                raise RuntimeError("GPU is not available")

        return providers

    def timer(self, title):
        return Timer(title)

    def stepper(self, step):
        # We assume that we call this method only once
        # just before the first iteration.
        elapsed = time.time() - self.created
        LOG.info("Model initialisation: %s", seconds(elapsed))
        return Stepper(step, self.lead_time)

    def _datetimes(self, dates):
        date = self.date
        assert isinstance(date, int)
        if date <= 0:
            date = datetime.datetime.utcnow() + datetime.timedelta(days=date)
            date = date.year * 10000 + date.month * 100 + date.day

        time = self.time
        assert isinstance(time, int)
        if time < 100:
            time *= 100
        assert time in (0, 600, 1200, 1800), time

        lagged = self.lagged
        if not lagged:
            lagged = [0]

        result = []
        for basedate in dates:
            for lag in lagged:
                date = basedate + datetime.timedelta(hours=lag)
                result.append(
                    (
                        date.year * 10000 + date.month * 100 + date.day,
                        date.hour,
                    ),
                )

        return result

    def datetimes(self):
        if self.staging_dates:
            dates = []
            with open(self.staging_dates) as f:
                for line in f:
                    dates.append(datetime.datetime.fromisoformat(line.strip()))

            return self._datetimes(dates)

        date = self.date
        assert isinstance(date, int)
        if date <= 0:
            date = datetime.datetime.utcnow() + datetime.timedelta(days=date)
            date = date.year * 10000 + date.month * 100 + date.day

        time = self.time
        assert isinstance(time, int)
        if time < 100:
            time *= 100

        assert time in (0, 600, 1200, 1800), time

        full = datetime.datetime(
            date // 10000,
            date % 10000 // 100,
            date % 100,
            time // 100,
            time % 100,
        )
        return self._datetimes([full])

    def print_fields(self):
        param, level = self.param_level_pl
        print("Grid:", self.grid)
        print("Area:", self.area)
        print("Pressure levels:")
        print("   Levels:", level)
        print("   Params:", param)
        print("Single levels:")
        print("   Params:", self.param_sfc)

    def print_assets_list(self):
        for file in self.download_files:
            print(file)

    def _print_request(self, verb, request, file=sys.stdout):
        r = [verb]
        for k, v in request.items():
            if not isinstance(v, (list, tuple, set)):
                v = [v]
            v = [str(_) for _ in v]
            v = "/".join(v)
            r.append(f"{k}={v}")

        r = ",\n   ".join(r)
        print(r, file=file)
        print(file=file)

    @property
    def _requests_extra(self):
        if not self.requests_extra:
            return {}
        extra = [_.split("=") for _ in self.requests_extra.split(",")]
        extra = {a: b for a, b in extra}
        return extra

    def print_requests(self):
        requests = self._requests()

        if self.json:
            print(json.dumps(requests, indent=4))
            return

        for r in requests:
            self._print_request("retrieve", r)

    def _requests(self):
        result = []

        first = dict(
            target="input.grib",
            grid=self.grid,
            area=self.area,
        )
        first.update(self.retrieve)

        for date, time in self.datetimes():  # noqa F402
            param, level = self.param_level_pl

            r = dict(
                levtype="pl",
                levelist=level,
                param=param,
                date=date,
                time=time,
            )
            r.update(first)
            first = {}

            r.update(self._requests_extra)

            self.patch_retrieve_request(r)

            result.append(dict(**r))

            r.update(
                dict(
                    levtype="sfc",
                    param=self.param_sfc,
                )
            )
            r.pop("levelist", None)

            self.patch_retrieve_request(r)
            result.append(dict(**r))

        return result

    def patch_retrieve_request(self, request):
        # Overriden in subclasses if needed
        pass

    def peek_into_checkpoint(self, path):
        return peek(path)

    def parse_model_args(self, args):
        if args:
            raise NotImplementedError(f"This model does not accept arguments {args}")

    def provenance(self):
        from .provenance import gather_provenance_info

        return gather_provenance_info(self.asset_files)


def load_model(name, **kwargs):
    return available_models()[name].load()(**kwargs)


def available_models():
    result = {}
    for e in entrypoints.get_group_all("ai_models.model"):
        result[e.name] = e
    return result
