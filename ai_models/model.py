# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
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


class Stepper:
    def __init__(self, step, lead_time):
        self.step = step
        self.lead_time = lead_time
        self.start = time.time()
        self.last = self.start
        self.num_steps = lead_time // step
        LOG.info("Starting inference for %s steps (%sh).", self.num_steps, lead_time)

    def __enter__(self):
        return self

    def __call__(self, i, step):
        now = time.time()
        elapsed = now - self.start
        speed = (i + 1) / elapsed
        eta = (self.num_steps - i) / speed
        LOG.info(
            "Done %s out of %s in %s (%sh), ETA: %s.",
            i + 1,
            self.num_steps,
            seconds(now - self.last),
            step,
            seconds(eta),
        )
        self.last = now

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("Elapsed: %s.", seconds(elapsed))
        LOG.info("Average: %s per step.", seconds(elapsed / self.num_steps))


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

    def __init__(self, input, output, download_assets, **kwargs):
        self.input = get_input(input, self, **kwargs)
        self.output = get_output(output, self, **kwargs)

        for k, v in kwargs.items():
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

        return device

    @cached_property
    def providers(self):
        import platform

        import GPUtil
        import onnxruntime as ort

        providers = []

        if GPUtil.getAvailable():
            providers += [
                "CUDAExecutionProvider",  # CUDA
            ]

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
        return providers

    def timer(self, title):
        return Timer(title)

    def stepper(self, step):
        return Stepper(step, self.lead_time)

    def datetimes(self):
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

        full = datetime.datetime(
            date // 10000,
            date % 10000 // 100,
            date % 100,
            time // 100,
            time % 100,
        )

        result = []
        for lag in lagged:
            date = full + datetime.timedelta(hours=lag)
            result.append(
                (
                    date.year * 10000 + date.month * 100 + date.day,
                    date.hour,
                ),
            )

        return result

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
        first = dict(
            target="input.grib",
            grid=self.grid,
            area=self.area,
        )
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

            self._print_request("retrieve", r)

            r = dict(
                levtype="sfc",
                param=self.param_sfc,
            )

            self._print_request("retrieve", r)

    def peek_into_checkpoint(self, path):
        return peek(path)


def load_model(name, **kwargs):
    return available_models()[name].load()(**kwargs)


def available_models():
    result = {}
    for e in entrypoints.get_group_all("ai_models.model"):
        result[e.name] = e
    return result
