import logging
import os
import sys
import time
from urllib.parse import urljoin

import requests
from multiurl import download
from multiurl import robust
from tqdm import tqdm

from .config import API_URL
from .config import CONFIG_PATH
from .config import load_config

LOG = logging.getLogger(__name__)


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
        config = load_config()

        self.url = url or os.getenv("AI_MODELS_REMOTE_URL") or config.get("url") or API_URL
        if not self.url.endswith("/"):
            self.url += "/"

        self.token = token or os.getenv("AI_MODELS_REMOTE_TOKEN") or config.get("token")

        if not self.token:
            LOG.error(
                "Missing remote token. Set it in %s or env AI_MODELS_REMOTE_TOKEN",
                CONFIG_PATH,
            )
            sys.exit(1)

        LOG.info("Using remote server %s", self.url)

        self.auth = BearerAuth(self.token)
        self.output_file = output_file
        self.input_file = input_file
        self._timeout = 300

    def run(self, cfg: dict):
        # upload file
        with open(self.input_file, "rb") as file:
            LOG.info("Uploading input file to remote server")
            data = self._request(requests.post, "upload", data=file)

        if data["status"] != "success":
            LOG.error(data["status"])
            if reason := data.get("reason"):
                LOG.error(reason)
            sys.exit(1)

        # submit task
        data = self._request(requests.post, data["href"], json=cfg)

        LOG.info("Inference request submitted")

        if data["status"] != "queued":
            LOG.error(data["status"])
            if reason := data.get("reason"):
                LOG.error(reason)
            sys.exit(1)

        LOG.info("Request id: %s", data["id"])
        LOG.info("Request is queued")

        last_status = data["status"]
        pbar = None

        while True:
            data = self._request(requests.get, data["href"])

            if data["status"] == "ready":
                if pbar is not None:
                    pbar.close()
                LOG.info("Request is ready")
                break

            if data["status"] == "failed":
                LOG.error("Request failed")
                if reason := data.get("reason"):
                    LOG.error(reason)
                sys.exit(1)

            if data["status"] != last_status:
                LOG.info("Request is %s", data["status"])
                last_status = data["status"]

            if progress := data.get("progress"):
                if pbar is None:
                    pbar = tqdm(
                        total=progress.get("total", 0),
                        unit="steps",
                        ncols=70,
                        leave=False,
                        initial=1,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}{postfix}",
                    )
                if eta := progress.get("eta"):
                    pbar.set_postfix_str(f"ETA: {eta}")
                if status := progress.get("status"):
                    pbar.set_description(status.strip().capitalize())
                pbar.update(progress.get("step", 0) - pbar.n)

            time.sleep(5)

        download(urljoin(self.url, data["href"]), target=self.output_file)

        LOG.debug("Result written to %s", self.output_file)

    def metadata(self, model, model_version, param) -> dict:
        if isinstance(param, str):
            return self._request(requests.get, f"metadata/{model}/{model_version}/{param}")
        elif isinstance(param, (list, dict)):
            return self._request(requests.post, f"metadata/{model}/{model_version}", json=param)
        else:
            raise ValueError("param must be a string, list, or dict with 'param' key.")

    def models(self):
        results = self._request(requests.get, "models")

        if not isinstance(results, list):
            return []

        return results

    def patch_retrieve_request(self, cfg, request):
        cfg["patchrequest"] = request
        result = self._request(requests.post, "patch", json=cfg)
        if status := result.get("status"):
            LOG.error(status)
            sys.exit(1)
        return result

    def _request(self, type, href, data=None, json=None, auth=None):
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

        try:
            data = response.json()

            if isinstance(data, dict) and (status := data.get("status")):
                data["status"] = status.lower()

            return data
        except Exception:
            return {"status": f"{response.url} {response.status_code} {response.text}"}
