from __future__ import annotations
from dataclasses import Field, dataclass, field, fields
from typing import Any, List
import sys
import argparse
from IPython.core.getipython import get_ipython

# see https://www.geeksforgeeks.org/how-to-add-colour-to-text-python/


def color(test_str, ansi_code):
    return f"\33[{ansi_code}m{test_str}\33[0m"


# GLOBALS
YELLOW, BLUE, PURPLE = 33, 34, 35

HEADER = f"""\
{'-' * 40}HyperParams{'-' * 40}
{' ' * 25}(color code: \
{color('default', BLUE)}, \
{color('config',  PURPLE)}, \
{color('command_line', YELLOW)})\
{' ' * 30}
"""

FOOTER = f"{'-' * 90}\n"
FILE_VARS = {}
COLOR_MAPPING = {}


def GET_COLOR(x) -> str:
    return color(x, COLOR_MAPPING.get(x, BLUE))


def induce_bool(value):
    if value.lower() in ("yes", "true", "t", "y"):
        return True
    elif value.lower() in ("no", "false", "f", "n"):
        return False
    else:
        raise ValueError


def add_to_parser(parser, name, val):
    if isinstance(val, bool):
        parser.add_argument(f"--{name}", type=induce_bool, default=val)
    else:
        parser.add_argument(f"--{name}", type=type(val), default=val)


def filter_cmdline_args(arg):
    return arg.startswith("--") and not arg.startswith("--f")


def read_config(file):
    try:
        variables = {}
        with open(file, encoding="utf-8") as f:
            exec(f.read(), variables)  # pylint: disable=exec-used
        return {k: v for k, v in variables.items() if not k.startswith("_")}
    except Exception as exc:
        raise ValueError(f"{file} is not a valid argument.") from exc


def TBD():
    return field(init=False, repr=False)  # pylint: disable=invalid-field-call


def load_globals(file_vars, changed_args):
    # pylint: disable=global-statement
    # global so not added as attr
    cmd_line_args = [x[0] for x in changed_args]
    color_mapping = {arg: PURPLE for var in file_vars.values() for arg in var.keys()}
    color_mapping.update({arg: YELLOW for arg in cmd_line_args})

    global COLOR_MAPPING
    global FILE_VARS
    FILE_VARS = file_vars
    COLOR_MAPPING = color_mapping


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


@dataclass
class Hypers:
    def __post_init__(self) -> None:
        self._raise_untyped()

        # set params in order default -> config -> cmd line
        # set default vars
        argv, changed_args = self._init_argsparse()

        file_vars = self.parse_config_files(argv)

        load_globals(file_vars, changed_args)

        # set config vars
        for variables in file_vars.values():
            for name, value in variables.items():
                self.set(name, value)

        # set cmd line vars
        for k, v in changed_args:
            self.set(k, v)

    def _init_argsparse(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        for f in self._all_fields():
            name, val = f.name, self.get(f.name)
            add_to_parser(parser, name, val)

        args, argv = parser.parse_known_args()
        keys = [arg.replace("--", "").split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")]
        changed_args = [(k, getattr(args, k)) for k in keys if k in args]

        return argv, changed_args

    def __str__(self):
        files = [f"- Reading {len(vars)} arguments from {file}\n" for file, vars in FILE_VARS.items()]
        args = [f"{GET_COLOR(k)}: {v}\n" for k, v in self.__dict__.items()]
        return "".join([HEADER, *files, *args, FOOTER])

    def _all_variables(self):
        return [n for n in self.__class__.__dict__ if not n.startswith("_")]

    def _all_fields(self) -> List[Field[Any]]:
        return list(filter(lambda f: f.init, fields(self)))

    def _raise_untyped(self):
        all_vars = set(self._all_variables())
        all_fields = {f.name for f in self._all_fields()}
        untyped_vars = all_vars - all_fields
        if len(untyped_vars) != 0:
            raise ValueError(f"Variables missing type annotations: {', '.join(untyped_vars)}")

    def get(self, name):
        return getattr(self, name)

    def set(self, name, val):
        setattr(self, name, val)

    def to_dict(self):
        return dict(self.__dict__)

    def update(self, new_dict):
        self.__dict__.update(new_dict)

    def parse_config_files(self, argv):
        configs = [f for f in argv if f.endswith(".py")]
        other = [f for f in argv if not f.endswith(".py")]

        if other:
            print(f"Skipping: {other}")

        return {} if is_notebook() else {file: read_config(file) for file in configs}
