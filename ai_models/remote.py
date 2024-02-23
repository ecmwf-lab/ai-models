import logging
import sys
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
        self._timeout = 300

    def run(self, cfg: dict, model_args: list):
        cfg.pop("remote_url", None)
        cfg.pop("remote_token", None)
        cfg["model_args"] = model_args

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
        except:
            status = f"{response.status_code} {response.url} {response.text}"
            href = None

        return status, href
