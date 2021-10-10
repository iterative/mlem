import json
import logging
import os
from typing import Dict

from opencensus.ext.azure.log_exporter import AzureEventHandler
import requests
from appdirs import user_config_dir
from filelock import FileLock, Timeout

from mlem import CONFIG
from mlem.log import logger
from mlem.version import __version__

# TOKEN = "s2s.jrn60wwvy9q6rc5r141pd.o723xpo21knm0xx4g4ppm"
# URL = "https://t.jitsu.com/api/v1/s2s/event"
CONNECTION_STRING = "InstrumentationKey=9e759683-de30-40d8-818f-08cd93f82144;IngestionEndpoint=https://westus2-2.in.applicationinsights.azure.com/"

azure_logger = logging.getLogger("mlem_telemetry")
azure_logger.setLevel(logging.INFO)
azure_logger.addHandler(AzureEventHandler(connection_string=CONNECTION_STRING))

def send_cli_call(cmd_name: str, **kwargs):
    send_event("cli", cmd_name, **kwargs)


def send_event(event_type: str, event_name: str, **kwargs):
    send({"event_type": event_type, "event_name": event_name, **kwargs})


def send(payload: Dict[str, str]):
    if not is_enabled():
        return
    payload.update(_runtime_info())
    try:
        azure_logger.info(payload['event_name'], extra={"custom_dimensions":payload})
        # requests.post(URL, params={"token": TOKEN}, json=payload, timeout=2)
    except Exception:  # noqa
        logger.debug("failed to send analytics report", exc_info=True)


def is_enabled():
    enabled = not CONFIG.TESTS and not CONFIG.NO_ANALYTICS

    logger.debug("Analytics is {}abled.".format("en" if enabled else "dis"))

    return enabled


def _runtime_info():
    """
    Gather information from the environment where DVC runs to fill a report.
    """

    return {
        "mlem_version": __version__,
        # "scm_class": _scm_in_use(),
        "system_info": _system_info(),
        "user_id": _find_or_create_user_id(),
    }


def _system_info():
    import platform
    import sys

    import distro

    system = platform.system()

    if system == "Windows":
        version = sys.getwindowsversion()

        return {
            "os": "windows",
            "windows_version_build": version.build,
            "windows_version_major": version.major,
            "windows_version_minor": version.minor,
            "windows_version_service_pack": version.service_pack,
        }

    if system == "Darwin":
        return {"os": "mac", "mac_version": platform.mac_ver()[0]}

    if system == "Linux":
        return {
            "os": "linux",
            "linux_distro": distro.id(),
            "linux_distro_like": distro.like(),
            "linux_distro_version": distro.version(),
        }

    # We don't collect data for any other system.
    raise NotImplementedError


def _find_or_create_user_id():
    """
    The user's ID is stored on a file under the global config directory.
    The file should contain a JSON with a "user_id" key:
        {"user_id": "16fd2706-8baf-433b-82eb-8c7fada847da"}
    IDs are generated randomly with UUID.
    """
    import uuid

    config_dir = user_config_dir("mlem", "Iterative")
    fname = os.path.join(config_dir, "user_id")
    lockfile = os.path.join(config_dir, "user_id.lock")

    # Since the `fname` and `lockfile` are under the global config,
    # we need to make sure such directory exist already.
    os.makedirs(config_dir, exist_ok=True)

    try:
        with FileLock(lockfile, timeout=5):
            try:
                with open(fname) as fobj:
                    user_id = json.load(fobj)["user_id"]

            except (FileNotFoundError, ValueError, KeyError):
                user_id = str(uuid.uuid4())

                with open(fname, "w") as fobj:
                    json.dump({"user_id": user_id}, fobj)

            return user_id

    except Timeout:
        logger.debug(f"Failed to acquire '{lockfile}'")