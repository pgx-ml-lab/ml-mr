import sys
import argparse
from .estimation.cli import (
    main as estimation_main
)


def main():
    """Main entry point for ml-mr.

    Example:

        ml-mr estimation --algorithm bin_iv -h

    """

    if len(sys.argv) < 3 or ("mode" != sys.argv[1]):
        return print_usage()

    mode = sys.argv[2]

    if mode == "estimation":
        return estimation_main()

    else:
        print(
            "'estimation' is currently the only supported mode.",
            file=sys.stderr
        )
        return print_usage()


def print_usage():
    print(
        "usage: ml-mr mode [-h] {estimation,evaluation,simulation}",
        file=sys.stderr
    )
    sys.exit(1)
