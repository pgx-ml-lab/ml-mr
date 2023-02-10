"""Logging utilities."""


import sys


def warn(message):
    print(message, file=sys.stderr)
