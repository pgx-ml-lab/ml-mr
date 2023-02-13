"""
Command-line interface entry-point for all tasks related to estimation.

This is mostly a dispatch into the implemented algorithms.

"""

import sys
import argparse

from .bin_iv import (
    configure_argparse as bin_iv_configure_argparse,
    main as bin_iv_main
)


def main():
    """Entry point for the estimation module.

    This is basically just a dispatch to specific algorithms.

    """
    parser = argparse.ArgumentParser(
        prog="ml-mr estimation"
    )

    algorithms = parser.add_subparsers(
        title="algorithm", dest="algorithm", required=True
    )

    bin_iv_parser = algorithms.add_parser("bin_iv")
    bin_iv_configure_argparse(bin_iv_parser)

    args = parser.parse_args(sys.argv[2:])
    if args.algorithm == "bin_iv":
        bin_iv_main(args)
    else:
        raise ValueError("Invalid algorithm.")
