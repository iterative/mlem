from rich.align import Align
from rich.console import Console
from rich.table import Column, Table
from rich.text import Text

from mlem.config import CONFIG

console = Console()


def echo(message):
    console.print(message)


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
    if not CONFIG.EMOJIS:
        return Text("")
    return Text(name + " ")


EMOJI_IMPORT = emoji("⤵️")
EMOJI_FAIL = emoji("❌")
