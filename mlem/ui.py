import contextlib
from typing import Callable, Optional

from rich.align import Align
from rich.console import Console
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from mlem.config import LOCAL_CONFIG

console = Console()
error_console = Console(stderr=True)

_echo_func: Optional[Callable] = None
_offset: int = 0


@contextlib.contextmanager
def set_echo(echo_func=...):
    global _echo_func  # pylint: disable=global-statement
    if echo_func is ...:
        yield
        return
    tmp = _echo_func
    try:
        _echo_func = echo_func
        yield
    finally:
        _echo_func = tmp


@contextlib.contextmanager
def set_offset(offset=0):
    global _offset  # pylint: disable=global-statement
    tmp = _offset
    try:
        _offset = offset
        yield
    finally:
        _offset = tmp


@contextlib.contextmanager
def cli_echo():
    with set_echo(console.print):
        yield


@contextlib.contextmanager
def stderr_echo():
    with set_echo(error_console.print):
        yield


@contextlib.contextmanager
def no_echo():
    with set_echo(None):
        yield


def echo(*message):
    if _offset > 0:
        message = [" " * (_offset - 1), *message]
    if _echo_func is not None:
        _echo_func(*message)


def boxify(text, col="red"):
    table = Table(
        Column(justify="center"),
        show_header=False,
        padding=(1, 4, 1, 4),
        style=col,
    )
    table.add_row(Align(text, align="center"))
    return table


def color(text, col):
    t = Text(text)
    t.stylize(col)
    return t


def emoji(name):
    if not LOCAL_CONFIG.EMOJIS:
        return Text("")
    return Text(name + " ")


def bold(text):
    return Style(bold=True).render(text)


EMOJI_LOAD = emoji("⏳️")
EMOJI_FAIL = emoji("❌")
EMOJI_OK = emoji("✅ ")
EMOJI_MLEM = emoji("🐶")
EMOJI_SAVE = emoji("💾")
EMOJI_APPLY = emoji("🍏")
EMOJI_COPY = emoji("🐏")
EMOJI_BASE = emoji("🏛")
EMOJI_NAILS = emoji("🖇️ ")
EMOJI_LINK = emoji("🔗")
EMOJI_PACK = emoji("💼")
EMOJI_BUILD = emoji("🛠")
EMOJI_UPLOAD = emoji("🔼")
EMOJI_STOP = emoji("🔻")
EMOJI_KEY = emoji("🗝")
