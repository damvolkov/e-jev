"""ejev entrypoint: the command tree is two tables, and nothing under commands/ loads until it runs.

`GROUPS` and `CARDS` are the whole surface — the e-cli shape. `build` turns them into cyclopts apps and
mounts every card by its `module:function` string, so `ejev --help` loads no command. The meta app runs the
`Launcher` first on every call: global options, the run's lifespan, injection by type, the error boundary.

    ejev noul · choice · score · ask · models · calibrate
"""

import sys
from importlib.metadata import version

from cyclopts import App

from e_jev.cli.runtime.launcher import Launcher
from e_jev.models.command import CommandCard, CommandGroup

##### TREE #####
GROUPS: tuple[CommandGroup, ...] = ()

CARDS: tuple[CommandCard, ...] = (
    CommandCard(path="noul", target="e_jev.cli.commands.decide:noul", summary="Yes/no: the probability of yes."),
    CommandCard(path="choice", target="e_jev.cli.commands.decide:choice", summary="Pick one of 2-255 options."),
    CommandCard(path="score", target="e_jev.cli.commands.decide:score", summary="Rate against 2-10 ordered levels."),
    CommandCard(path="ask", target="e_jev.cli.commands.decide:ask", summary="Send a full request body from a JSON file."),
    CommandCard(path="models", target="e_jev.cli.commands.decide:listing", summary="Models the endpoint advertises."),
    CommandCard(path="calibrate", target="e_jev.cli.commands.calibrate:calibrate", summary="Fit the temperature (run in jev)."),
)


def build() -> App:
    """Composition root: the tree from the tables, the launcher on the meta app. Imports no command module."""
    root = App(
        name="ejev", help="Ask Jev — or e-jev — typed questions from the terminal.", version=version("e-jev"), result_action="return_value"
    )
    apps = {"": root}
    for group in GROUPS:
        apps[group.path] = App(name=group.name, help=group.help)
        apps[group.parent].command(apps[group.path])
    for card in CARDS:
        apps[card.parent].command(card.target, name=card.name, help=card.summary)
    root.meta.default(Launcher(root, CARDS).run)
    return root


def main() -> None:
    """The `ejev` script: build, run, exit with the launcher's code."""
    sys.exit(build().meta(sys.argv[1:]))
