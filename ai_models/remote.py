import logging
import time
from urllib.parse import urljoin

import requests
from multiurl import download, robust

LOG = logging.getLogger(__name__)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class RemoteClient:
    def __init__(self, url: str, token: str, output_file: str, input_file: str = None):
        self.url = url
        self.auth = BearerAuth(token)
        self.output_file = output_file
        self.input_file = input_file
        self._timeout = 10
        self._status = self._last_status = None

    def run(self, cfg: dict, model_args: list):
        cfg.pop("remote_url", None)
        cfg.pop("remote_token", None)
        cfg["model_args"] = model_args

        if self._upload(self.input_file) == "success":
            self._submit(cfg)

        while not self._ready():
            time.sleep(5)

        download(urljoin(self.url, self._href), target=self.output_file)

        LOG.debug("Result written to %s", self.output_file)

    def _upload(self, file):
        with open(file, "rb") as f:
            LOG.info("Uploading input file to remote")
            r = robust(requests.post, retry_after=self._timeout)(
                urljoin(self.url, "upload"),
                data=f,
                auth=self.auth,
                timeout=self._timeout,
            ).json()

        LOG.debug(r)
        self._uid = r["id"]
        self._href = r["href"]
        self._status = r["status"].lower()

        return self._status

    def _submit(self, data):
        r = robust(requests.post, retry_after=self._timeout)(
            urljoin(self.url, self._href),
            json=data,
            auth=self.auth,
            timeout=self._timeout,
        ).json()

        LOG.debug(r)
        self._uid = r["id"]
        self._href = r["href"]
        self._status = r["status"].lower()

        return self._status

    def _poll(self):
        r = robust(requests.get, retry_after=self._timeout)(
            urljoin(self.url, self._href), auth=self.auth, timeout=self._timeout
        ).json()

        LOG.debug(r)
        self._href = r["href"]
        self._status = r["status"].lower()

        if self._last_status != self._status:
            LOG.info("Job status: %s", self._status)
            self._last_status = self._status

        return self._status

    def _ready(self):
        return self._poll() == "ready"
