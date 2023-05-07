import sys
from .estimation.cli import main as estimation_main
from .evaluation.cli import main as evaluation_main
from .sweep.cli import main as sweep_main


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

    elif mode == "evaluation":
        return evaluation_main()

    elif mode == "sweep":
        return sweep_main()

    else:
        print(
            f"Unknown mode '{mode}'.",
            file=sys.stderr
        )
        return print_usage()


def print_usage():
    print(
        "usage: ml-mr {estimation,evaluation,sweep} [-h]",
        file=sys.stderr
    )
    sys.exit(1)
