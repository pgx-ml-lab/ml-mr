"""
Command-line interface entry-point for all tasks related to model evaluation.

# ml-mr evaluate
#     --input path \
#     --true-function 'my_file.py:true_function' \

"""

from typing import Tuple, Callable
import sys
import os
import json
import argparse

import torch
import numpy as np

from ..estimation import MODELS, MREstimator, MREstimatorWithUncertainty
from .metrics import mse, mean_prediction_interval_absolute_width
from ..logging import warn, info


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
        "--true-function",
        required=True,
        type=str,
        help="Filename and name of a python script and the function "
             "representing the true causal effect. For example: "
             "'my_file.py:function_name'.\n"
             "Alternatively, this can be a lambda expression."
    )

    parser.add_argument(
        "--domain",
        default=None,
        type=str,
        help="Domain used to evaluate metrics such as the mean squared error."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the true function and prediction together."
    )

    return parser.parse_args(argv)


def plot(
    estimator: MREstimator,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float] = (-3, 3),
    label: str = "Predicted Y",
    plot_structural: bool = True,
    n_points: int = 5000
):
    import matplotlib.pyplot as plt

    xs = torch.linspace(domain[0], domain[1], n_points)

    uncertainty = False
    if isinstance(estimator, MREstimatorWithUncertainty):
        uncertainty = True
        y_hat_ci = estimator.effect_with_prediction_interval(xs, alpha=0.1)
        y_hat_l = y_hat_ci[:, 0]
        y_hat = y_hat_ci[:, 1]
        y_hat_u = y_hat_ci[:, 2]
    else:
        y_hat = estimator.effect(xs)

    true_y = true_function(xs)

    if plot_structural:
        plt.plot(
            xs.numpy(),
            true_y.numpy().reshape(-1),
            ls="--",
            color="#9C0D00",
            lw=1,
            label="True Y"
        )

    plt.scatter(
        xs.numpy(),
        y_hat.numpy().reshape(-1),
        s=1,
        label=label
    )
    if uncertainty:
        plt.fill_between(
            xs.numpy(),
            y_hat_l.numpy().reshape(-1),
            y_hat_u.numpy().reshape(-1),
            zorder=-1,
            color="#eeeeee"
        )


def main():
    args = parse_args(sys.argv[2:])

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

    if args.plot:
        # Setup figure.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))

    # Load MREstimator.
    n_plotted = 0
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
            estimator, true_function, domain=(domain_lower, domain_upper)
        )

        print(input, cur_mse, sep=",", end="")

        if isinstance(estimator, MREstimatorWithUncertainty):
            width = mean_prediction_interval_absolute_width(
                estimator, [domain_lower, domain_upper], 0.1
            )
            print(f",{width}", end="")

        print()

        if args.plot:
            if n_plotted == 11:
                warn("Not plotting more than 10 curves in batch mode.")
            elif n_plotted > 11:
                pass
            else:
                plot(
                    estimator,
                    true_function,
                    domain=(domain_lower, domain_upper),
                    label=input,
                    plot_structural=True if n_plotted == 0 else False
                )
                n_plotted += 1

    if args.plot:
        # Finalize and show figure.
        plt.legend()
        plt.tight_layout()
        plt.show()
