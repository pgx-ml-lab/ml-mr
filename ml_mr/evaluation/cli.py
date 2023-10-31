"""
Command-line interface entry-point for all tasks related to model evaluation.

# ml-mr evaluate
#     --input path \
#     --true-function 'my_file.py:true_function' \

"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

from ..estimation import MODELS, MREstimatorWithUncertainty
from ..logging import warn
from .metrics import (
    mean_coverage,
    mean_prediction_interval_absolute_width,
    mse
)
from .plotting import plot_iv_reg


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="ml-mr evaluation"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to an estimated model(s).",
        nargs="+"
    )

    parser.add_argument(
        "--no-header",
        action="store_true"
    )

    parser.add_argument(
        "--true-function",
        required=True,
        type=str,
        help="Filename and name of a python script and the function "
             "representing the true causal effect. For example: "
             "'my_file.py:function_name'.\n"
             "Alternatively, this can be a lambda expression."
    )

    parser.add_argument(
        "--domain-95",
        action="store_true",
        help="Use the 2.5 and 97.5th percentiles of the training exposure "
             "distribution as the domain."
    )

    parser.add_argument(
        "--domain",
        default=None,
        type=str,
        help="Domain used to evaluate metrics such as the mean squared error."
    )

    parser.add_argument(
        "--sample-n-covars",
        default=10_000,
        help="Sample covariable when computing ATEs to save on computation."
    )

    parser.add_argument(
        "--meta-keys",
        nargs="*",
        help="Keys to extract from the meta.json file of the fit. It will be "
             "printed in CSV format.",
        default=[]
    )

    parser.add_argument(
        "--plot-max-lines",
        type=int,
        default=5,
        help="Maximum number of estimates shown on the plot."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the true function and prediction together."
    )

    parser.add_argument(
        "--plot-filename",
        default=None,
        type=str,
        help="Output filename to save the plot to disk."
    )

    parser.add_argument(
        "--alpha",
        default=0.1,
        type=float,
        help="Alpha (miscoverage) level for prediction intervals and metrics."
    )

    args = parser.parse_args(argv)

    if args.domain is not None and args.domain_95:
        raise ValueError(
            "Provide either an explicit domain or use --domain-95."
        )

    return args


def main():
    args = parse_args(sys.argv[2:])
    writer = csv.writer(sys.stdout)

    # Load the true function.
    if args.true_function.startswith("lambda"):
        # This is not safe, but it's very useful.
        true_function = eval(
            args.true_function,
            {"torch": torch, "np": np}
        )
    else:
        filename, function_name = args.true_function.split(":")

        splitted = os.path.split(filename)
        if len(splitted) == 2:
            # We add the module's directory to Python path.
            sys.path.append(splitted[0])
            filename = splitted[1]

        true_function = getattr(
            __import__(filename.replace(".py", "")),
            function_name
        )

    # Parse domain.
    if args.domain is not None:
        domain_lower, domain_upper = [float(i) for i in args.domain.split(",")]
    else:
        domain_lower = None
        domain_upper = None

    ax = None
    if args.plot:
        # Setup figure.
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Header
    if not args.no_header:
        header = [
            "filename", "mse", "mean_pred_interval_width", "mean_coverage"
        ]
        header.extend(args.meta_keys)
        writer.writerow(header)

    n_plotted = 0
    n_input = len(args.input)
    legend_lines = []
    for input in args.input:
        # Try to detect model type.
        meta_filename = os.path.join(input, "meta.json")
        try:
            with open(meta_filename, "rt") as f:
                meta = json.load(f)
        except FileNotFoundError:
            warn(
               f"Could not find metadata for ml-mr fitted model in "
               f"'{input}'. Ignoring."
            )
            continue

        if args.domain_95:
            domain_lower, domain_upper = meta["exposure_95_percentile"]

        meta_values = []
        if args.meta_keys:
            meta_values = [
                str(meta.get(key, "")) for key in args.meta_keys
            ]

        # Set domain if it wasn't set explicitly.
        if domain_lower is None:
            assert domain_upper is None
            domain_lower, domain_upper = meta["domain"]

        loader = MODELS[meta["model"]]["load"]

        try:
            estimator = loader(input)
        except FileNotFoundError:
            warn(f"Couldn't load model '{input}'. Ignoring.")
            continue

        cur_mse = mse(
            estimator, true_function, domain=(domain_lower, domain_upper),
        )

        row = [input, cur_mse]
        if isinstance(estimator, MREstimatorWithUncertainty):
            width = mean_prediction_interval_absolute_width(
                estimator, (domain_lower, domain_upper),
                alpha=args.alpha
            )
            row.append(width)

            coverage = mean_coverage(
                estimator, true_function, (domain_lower, domain_upper),
            )
            row.append(coverage)

        else:
            row.extend(["", ""])  # No prediction interval width and coverage

        row.extend(meta_values)
        writer.writerow(row)

        if args.plot:
            max_plots = args.plot_max_lines
            if n_plotted == max_plots:
                warn(
                    f"Not plotting more than {max_plots} curves in batch mode."
                )
                n_plotted += 1  # To avoid printing warning multiple times.
            elif n_plotted > max_plots:
                pass
            else:
                plot_data = plot_iv_reg(
                    estimator,
                    true_function,
                    domain=(domain_lower, domain_upper),
                    label=input.lstrip("_"),  # Otherwise ignored by MPL
                    plot_structural=True if n_plotted == 0 else False,
                    alpha=args.alpha,
                    ax=ax,
                    multi_run=n_input > 1
                )
                ax = plot_data["ax"]
                legend_lines.append(plot_data["lines"])
                n_plotted += 1

    if args.plot:
        # Finalize and show figure.
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(handles=legend_lines, prop={"size": 14})
        plt.tight_layout()

        if args.plot_filename is None:
            plt.show()
        else:
            if args.plot_filename.endswith(".png"):
                plt.savefig(args.plot_filename, dpi=800)
            else:
                plt.savefig(args.plot_filename)
