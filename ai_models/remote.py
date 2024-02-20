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


class RemoteRunner:
    def __init__(self, url: str, token: str, output_file: str, input_file: str = None):
        self.url = url
        self.auth = BearerAuth(token)
        self.output_file = output_file
        self.input_file = input_file
        self._timeout = 10

    def run(self, cfg: dict, model_args: list):
        cfg.pop("remote_url", None)
        cfg.pop("remote_token", None)
        cfg["model_args"] = model_args

        r = self._submit(cfg)
        status = self._last_status
        LOG.debug(r)
        LOG.info("Job status: %s", self._last_status)

        while not self._ready():
            if status != self._last_status:
                status = self._last_status
                LOG.info("Job status: %s", status)
            time.sleep(5)

        download(urljoin(self.url, self._href), target=self.output_file)

        LOG.info("Result written to %s", self.output_file)

    def _submit(self, data):
        r = robust(requests.post, retry_after=self._timeout)(
            urljoin(self.url, "submit"),
            json=data,
            auth=self.auth,
            timeout=self._timeout,
        )
        res = r.json()
        self._uid = res["id"]
        self._href = res["href"]
        self._last_status = res["status"]
        return res

    def _status(self):
        r = robust(requests.get, retry_after=self._timeout)(
            urljoin(self.url, self._href), auth=self.auth, timeout=self._timeout
        )

        res = r.json()
        LOG.debug(res)
        self._href = res["href"]
        self._last_status = res["status"]

        return self._last_status

    def _ready(self):
        return self._status().lower() == "ready"
