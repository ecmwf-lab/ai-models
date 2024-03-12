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
        kwargs["download_assets"] = False

        super().__init__(**kwargs)

        self.cfg = kwargs
        self.client = RemoteClient()

    def run(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_file = os.path.join(tmpdirname, "input.grib")
            output_file = os.path.join(tmpdirname, "output.grib")
            self.all_fields.save(input_file)

            self.client.input_file = input_file
            self.client.output_file = output_file

            self.client.run(self.cfg)

            ds = cml.load_source("file", output_file)
            for field in ds:
                self.write(None, template=field)

    def parse_model_args(self, args):
        return None


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class RemoteClient:
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
            LOG.info("Using remote %s", url)

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
        with open(self.input_file, "rb") as f:
            LOG.info("Uploading input file to remote")
            status, href = self._request(requests.post, "upload", data=f)

        if status != "success":
            LOG.error(status)
            sys.exit(1)

        # submit job
        status, href = self._request(requests.post, href, json=cfg)

        if status != "queued":
            LOG.error(status)
            sys.exit(1)

        LOG.info("Job status: queued")
        last_status = status

        while True:
            status, href = self._request(requests.get, href)

            if status != last_status:
                LOG.info("Job status: %s", status)
                last_status = status

            if status == "failed":
                sys.exit(1)

            if status == "ready":
                break

            time.sleep(4)

        download(urljoin(self.url, href), target=self.output_file)

        LOG.debug("Result written to %s", self.output_file)

    def _request(self, type, href, data=None, json=None, auth=None):
        r = robust(type, retry_after=self._timeout)(
            urljoin(self.url, href),
            json=json,
            data=data,
            auth=self.auth,
            timeout=self._timeout,
        )

        status, href = self._update_state(r)
        return status, href

    def _update_state(self, response: requests.Response):
        if response.status_code == 401:
            return "Unauthorized Access", None

        try:
            data = response.json()
            href = data["href"]
            status = data["status"].lower()
        except Exception:
            status = f"{response.status_code} {response.url} {response.text}"
            href = None

        return status, href
