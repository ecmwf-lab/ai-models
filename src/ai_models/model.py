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

import climetlab as cml
import entrypoints
import numpy as np
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
    UNIQUE = {"date", "hdate", "time", "referenceDate", "type", "stream", "expver"}

    def __init__(self) -> None:
        self.expect = 0
        self.request = defaultdict(set)

    def add(self, field):
        self.expect += 1
        for k, v in field.items():
            self.request[k].add(str(v))
            if k in self.UNIQUE:
                if len(self.request[k]) > 1:
                    raise ValueError(f"Field {field} has different values for {k}: {self.request[k]}")


class Model:
    lagged = False
    assets_extra_dir = None
    retrieve = {}  # Extra parameters for retrieve
    version = 1  # To be overriden in subclasses
    grib_extra_metadata = {}  # Extra metadata for grib files

    param_level_ml = ([], [])  # param, level
    param_level_pl = ([], [])  # param, level
    param_sfc = []  # param

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
    def fields_ml(self):
        return self.input.fields_ml

    @cached_property
    def fields_sfc(self):
        return self.input.fields_sfc

    @cached_property
    def all_fields(self):
        return self.input.all_fields

    def write(self, *args, **kwargs):
        self.collect_archive_requests(
            self.output.write(*args, **kwargs, **self.grib_extra_metadata),
        )

    def collect_archive_requests(self, written):
        if self.archive_requests:
            handle, path = written
            if self.hindcast_reference_year or self.hindcast_reference_date:
                # The clone is necessary because the handle
                # does not return always return recently set keys
                handle = handle.clone()

            self.archiving[path].add(handle.as_mars())

    def finalise(self):
        self.output.finalise()

        if self.archive_requests:
            with open(self.archive_requests, "w") as f:
                json_requests = []

                for path, archive in self.archiving.items():
                    request = dict(expect=archive.expect)
                    if path is not None:
                        request["source"] = f'"{path}"'
                    request.update(archive.request)
                    request.update(self._requests_extra)

                    if self.json:
                        json_requests.append(request)
                    else:
                        self._print_request("archive", request, file=f)

                if json_requests:

                    def json_default(obj):
                        if isinstance(obj, set):
                            if len(obj) > 1:
                                return list(obj)
                            else:
                                return obj.pop()
                        return obj

                    print(
                        json.dumps(json_requests, separators=(",", ":"), default=json_default),
                        file=f,
                    )

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

    def torch_deterministic_mode(self):
        import torch

        LOG.info("Setting deterministic mode for PyTorch")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    @cached_property
    def providers(self):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        providers = []
        for n in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            if n in available_providers:
                providers.append(n)

        LOG.info(
            "Using device '%s'. The speed of inference depends greatly on the device.",
            ort.get_device(),
        )

        if self.only_gpu:
            assert isinstance(ort.get_device(), str)
            if ort.get_device() == "CPU":
                raise RuntimeError("GPU is not available")

            providers = ["CUDAExecutionProvider"]

        LOG.info("ONNXRuntime providers: %s", providers)

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

    def datetimes(self, step=0):
        if self.staging_dates:
            assert step == 0, step
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
        ) + datetime.timedelta(hours=step)
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
        for k, v in sorted(request.items()):
            if not isinstance(v, (list, tuple, set)):
                v = [v]

            if k in ("area", "grid", "frame", "rotation", "bitmap"):
                v = [str(_) for _ in v]
            else:
                v = [str(_) for _ in sorted(v)]

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

    def _requests_unfiltered(self):
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

    def _requests(self):

        def filter_constant(request):
            # We check for 'sfc' because param 'z' can be ambiguous
            if request.get("levtype") == "sfc":
                param = set(self.constant_fields) & set(request.get("param", []))
                if param:
                    request["param"] = list(param)
                    return True

            return False

        def filter_prognostic(request):
            # TODO: We assume here that prognostic fields are
            # the ones that are not constant. This may not always be true
            if request.get("levtype") == "sfc":
                param = set(request.get("param", [])) - set(self.constant_fields)
                if param:
                    request["param"] = list(param)
                    return True
                return False

            return True

        def filter_last_date(request):
            date, time = max(self.datetimes())
            return request["date"] == date and request["time"] == time

        def noop(request):
            return request

        filter_type = {
            "constants": filter_constant,
            "prognostics": filter_prognostic,
            "all": noop,
        }[self.retrieve_fields_type]

        filter_dates = {
            True: filter_last_date,
            False: noop,
        }[self.retrieve_only_one_date]

        result = []
        for r in self._requests_unfiltered():
            if filter_type(r) and filter_dates(r):
                result.append(r)

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

    def forcing_and_constants(self, date, param):
        source = self.all_fields[:1]

        ds = cml.load_source(
            "constants",
            source,
            date=date,
            param=param,
        )

        assert len(ds) == len(param), (len(ds), len(param), date)

        return ds.to_numpy(dtype=np.float32)

    @cached_property
    def gridpoints(self):
        return len(self.all_fields[0].grid_points()[0])

    @cached_property
    def start_datetime(self):
        return self.all_fields.order_by(valid_datetime="ascending")[-1].datetime()

    @property
    def constant_fields(self):
        raise NotImplementedError("constant_fields")

    def write_input_fields(
        self,
        fields,
        accumulations=None,
        accumulations_template=None,
        accumulations_shape=None,
        ignore=None,
    ):
        if ignore is None:
            ignore = []

        with self.timer("Writing step 0"):
            for field in fields:
                if field.metadata("shortName") in ignore:
                    continue

                if field.valid_datetime() == self.start_datetime:
                    self.write(
                        None,
                        template=field,
                        step=0,
                    )

            if accumulations is not None:
                if accumulations_template is None:
                    accumulations_template = fields.sel(param="2t")[0]

                if accumulations_shape is None:
                    accumulations_shape = accumulations_template.shape

                for param in accumulations:
                    self.write(
                        np.zeros(accumulations_shape, dtype=np.float32),
                        stepType="accum",
                        template=accumulations_template,
                        param=param,
                        startStep=0,
                        endStep=0,
                        date=int(self.start_datetime.strftime("%Y%m%d")),
                        time=int(self.start_datetime.strftime("%H%M")),
                        check=True,
                    )


def load_model(name, **kwargs):
    return available_models()[name].load()(**kwargs)


def available_models():
    result = {}
    for e in entrypoints.get_group_all("ai_models.model"):
        result[e.name] = e
    return result
