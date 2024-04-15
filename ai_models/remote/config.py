import logging
import os

API_URL = "https://ai-models.ecmwf.int/api/v1/"

ROOT_PATH = os.path.join(os.path.expanduser("~"), ".config", "ai-models")
CONFIG_PATH = os.path.join(ROOT_PATH, "api.yaml")

LOG = logging.getLogger(__name__)


def config_exists():
    return os.path.exists(CONFIG_PATH)


def create_config():
    if config_exists():
        return

    try:
        os.makedirs(ROOT_PATH, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            f.write("token: \n")
            f.write(f"url: {API_URL}\n")
    except Exception as e:
        LOG.error(f"Failed to create config {CONFIG_PATH}")
        LOG.error(e, exc_info=True)


def load_config() -> dict:
    from yaml import safe_load

    if not config_exists():
        create_config()

    try:
        with open(CONFIG_PATH, "r") as f:
            return safe_load(f) or {}
    except Exception as e:
        LOG.error(f"Failed to read config {CONFIG_PATH}")
        LOG.error(e, exc_info=True)
        return {}
