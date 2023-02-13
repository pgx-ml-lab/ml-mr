import sys
from .estimation.cli import (
    main as estimation_main
)


def main():
    """Main entry point for ml-mr.

    Example:

        ml-mr estimation --algorithm bin_iv -h

    """

    if len(sys.argv) < 2:
        return print_usage()

    mode = sys.argv[1]

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
        "usage: ml-mr {estimation,evaluation,simulation} [-h]",
        file=sys.stderr
    )
    sys.exit(1)
