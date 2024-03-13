import logging
import os
import sys
import tempfile
import time
from functools import cached_property
from urllib.parse import urljoin

import climetlab as cml
import requests
from multiurl import download, robust

from .model import Model

LOG = logging.getLogger(__name__)


class RemoteModel(Model):
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.cfg["download_assets"] = False
        self.cfg["assets_extra_dir"] = None

        self.model = self.cfg["model"]
        self._param = {}
        self.api = RemoteAPI()

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

    def get_parameter(self, name):
        if (param := self._param.get(name, None)) is not None:
            return param

        self._param.update(self.api.metadata(self.model, name))

        return self._param[name]

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


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class RemoteAPI:
    def __init__(
        self,
        input_file: str = None,
        output_file: str = "output.grib",
        url: str = None,
        token: str = None,
    ):
        root = os.path.join(os.path.expanduser("~"), ".config", "ai-models")
        os.makedirs(root, exist_ok=True)

        configfile = os.path.join(root, "api.yaml")

        if os.path.exists(configfile):
            from yaml import safe_load

            with open(configfile, "r") as f:
                config = safe_load(f) or {}

                url = config.get("url", None)
                token = config.get("token", None)

        if url is None:
            url = os.getenv("AI_MODELS_REMOTE_URL", "https://ai-models.ecmwf.int")
            LOG.info("Using remote server %s", url)

        token = token or os.getenv("AI_MODELS_REMOTE_TOKEN", None)

        if token is None:
            LOG.error(
                "Missing remote token. Set it in %s or in env AI_MODELS_REMOTE_TOKEN",
                configfile,
            )
            sys.exit(1)

        self.url = url
        self.token = token
        self.auth = BearerAuth(token)
        self.output_file = output_file
        self.input_file = input_file
        self._timeout = 300

    def run(self, cfg: dict):
        # upload file
        with open(self.input_file, "rb") as file:
            LOG.info("Uploading input file to remote server")
            _, status, href = self._request(requests.post, "upload", data=file)

        if status != "success":
            LOG.error(status)
            sys.exit(1)

        # submit task
        id, status, href = self._request(requests.post, href, json=cfg)

        LOG.info("Request submitted")
        LOG.info("Request id: %s", id)

        if status != "queued":
            LOG.error(status)
            sys.exit(1)

        LOG.info("Request is queued")
        last_status = status

        while True:
            _, status, href = self._request(requests.get, href)

            if status != last_status:
                LOG.info("Request is %s", status)
                last_status = status

            if status == "failed":
                sys.exit(1)

            if status == "ready":
                break

            time.sleep(5)

        download(urljoin(self.url, href), target=self.output_file)

        LOG.debug("Result written to %s", self.output_file)

    def metadata(self, model, param) -> dict:
        if isinstance(param, str):
            return self._request(
                requests.get, f"metadata/{model}/{param}", with_status=False
            )
        elif isinstance(param, (list, dict)):
            return self._request(
                requests.post, f"metadata/{model}", json=param, with_status=False
            )
        else:
            raise ValueError("param must be a string, list, or dict with 'param' key.")

    def _request(self, type, href, data=None, json=None, auth=None, with_status=True):
        response = robust(type, retry_after=30)(
            urljoin(self.url, href),
            json=json,
            data=data,
            auth=self.auth,
            timeout=self._timeout,
        )

        if response.status_code == 401:
            LOG.error("Unauthorized Access. Check your token.")
            sys.exit(1)

        if with_status:
            id, status, href = self._update_state(response)
            return id, status, href
        else:
            return response.json()

    def _update_state(self, response: requests.Response):
        try:
            data = response.json()
            href = data["href"]
            status = data["status"].lower()
            id = data["id"]
        except Exception:
            status = f"{response.status_code} {response.url} {response.text}"
            href = None
            id = None

        return id, status, href
