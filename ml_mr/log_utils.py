"""Logging utilities."""


import sys


def warn(message):
    print(message, file=sys.stderr)


def critical(message):
    print("***", message, file=sys.stderr)


def info(message):
    print(message)


def debug(message):
    print(f"[DEBUG] {message}", file=sys.stderr)
