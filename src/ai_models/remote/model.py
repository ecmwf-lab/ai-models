import logging
import os
import sys
import tempfile
from functools import cached_property

import climetlab as cml

from ..model import Model
from .api import RemoteAPI

LOG = logging.getLogger(__name__)


class RemoteModel(Model):
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.cfg["download_assets"] = False

        self.model = self.cfg["model"]
        self.model_version = self.cfg.get("model_version", "latest")
        self._param = {}
        self.api = RemoteAPI()

        if self.model not in self.api.models():
            LOG.error(f"Model '{self.model}' not available on remote server.")
            LOG.error("Rerun the command with --models --remote to list available remote models.")
            sys.exit(1)

        self.load_parameters()

        super().__init__(**self.cfg)

    def __getattr__(self, name):
        return self.get_parameter(name)

    def run(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_file = os.path.join(tmpdirname, "input.grib")
            output_file = os.path.join(tmpdirname, "output.grib")
            self.all_fields.save(input_file)

            self.api.input_file = input_file
            self.api.output_file = output_file

            self.api.run(self.cfg)

            ds = cml.load_source("file", output_file)
            for field in ds:
                self.write(None, template=field)

    def parse_model_args(self, args):
        return None

    def patch_retrieve_request(self, request):
        if not self.remote_has_patch:
            return

        patched = self.api.patch_retrieve_request(self.cfg, request)
        request.update(patched)

    def load_parameters(self):
        params = self.api.metadata(
            self.model,
            self.model_version,
            [
                "expver",
                "version",
                "grid",
                "area",
                "param_level_ml",
                "param_level_pl",
                "param_sfc",
                "lagged",
                "grib_extra_metadata",
                "retrieve",
                "remote_has_patch",  # custom parameter, checks if remote model need patches
            ],
        )
        self._param.update(params)

    def get_parameter(self, name):
        if (param := self._param.get(name)) is not None:
            return param

        _param = self.api.metadata(self.model, self.model_version, name)
        self._param.update(_param)

        return self._param.get(name)

    @cached_property
    def param_level_ml(self):
        return self.get_parameter("param_level_ml") or ([], [])

    @cached_property
    def param_level_pl(self):
        return self.get_parameter("param_level_pl") or ([], [])

    @cached_property
    def param_sfc(self):
        return self.get_parameter("param_sfc") or []

    @cached_property
    def lagged(self):
        return self.get_parameter("lagged") or False

    @cached_property
    def version(self):
        return self.get_parameter("version") or 1

    @cached_property
    def grib_extra_metadata(self):
        return self.get_parameter("grib_extra_metadata") or {}

    @cached_property
    def retrieve(self):
        return self.get_parameter("retrieve") or {}
