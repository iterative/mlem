import json
import logging
import os
import subprocess
from threading import Thread
from typing import Dict

import requests
from appdirs import user_config_dir
from filelock import FileLock, Timeout

from mlem import CONFIG
from mlem.version import __version__

logger = logging.getLogger(__name__)
TOKEN = "s2s.9vdp1745vpkibkcxznsfus.cgsh70aoy3m39bfuey6shn"
URL = "https://telemetry.mlem.ai/api/v1/s2s/event?ip_policy=strict"


def send_cli_call(cmd_name: str, **kwargs):
    send_event("cli", cmd_name, **kwargs)


def send_event(
    event_type: str,
    event_name: str,
    use_thread: bool = False,
    use_daemon: bool = True,
    **kwargs,
):
    send(
        {"event_type": event_type, "event_name": event_name, **kwargs},
        use_thread=use_thread,
        use_daemon=use_daemon,
    )


def send(
    payload: Dict[str, str], use_thread: bool = False, use_daemon: bool = True
):
    if not is_enabled():
        return
    payload.update(_runtime_info())
    if use_thread and use_daemon:
        raise ValueError(
            "Both use_thread and use_daemon cannot be true at the same time"
        )
    impl = _send
    if use_daemon:
        impl = _send_daemon
    if use_thread:
        impl = _send_thread
    impl(payload)


def _send_daemon(payload):
    import sys

    cmd = f"import requests;requests.post('{URL}',params={{'token':'{TOKEN}'}},json={payload})"

    if os.name == "nt":

        from subprocess import (
            CREATE_NEW_PROCESS_GROUP,
            CREATE_NO_WINDOW,
            STARTF_USESHOWWINDOW,
            STARTUPINFO,
        )

        detached_flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        startupinfo = STARTUPINFO()
        startupinfo.dwFlags |= STARTF_USESHOWWINDOW
        subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", cmd],
            creationflags=detached_flags,
            close_fds=True,
            startupinfo=startupinfo,
        )
    elif os.name == "posix":
        subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", cmd],
            close_fds=True,
        )
    else:
        raise NotImplementedError


def _send_thread(payload):
    Thread(target=_send, args=[payload]).start()


def _send(payload):
    try:
        requests.post(URL, params={"token": TOKEN}, json=payload, timeout=2)
    except Exception:  # pylint: disable=broad-except
        logger.debug("failed to send analytics report", exc_info=True)


def is_enabled():
    enabled = not CONFIG.TESTS and not CONFIG.NO_ANALYTICS

    msg = f"Analytics are {'en' if enabled else 'dis'}abled."
    logger.debug(msg)

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
        version = sys.getwindowsversion()  # pylint: disable=no-member

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
        with FileLock(  # pylint: disable=abstract-class-instantiated
            lockfile, timeout=5
        ):
            try:
                with open(fname, encoding="utf8") as fobj:
                    user_id = json.load(fobj)["user_id"]

            except (FileNotFoundError, ValueError, KeyError):
                user_id = str(uuid.uuid4())

                with open(fname, "w", encoding="utf8") as fobj:
                    json.dump({"user_id": user_id}, fobj)

            return user_id

    except Timeout:
        logger.debug("Failed to acquire %s", lockfile)
    return None
