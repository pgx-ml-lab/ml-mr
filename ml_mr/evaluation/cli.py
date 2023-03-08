"""
Command-line interface entry-point for all tasks related to model evaluation.

# ml-mr evaluate
#     --input path \
#     --model bin_iv \
#     --true-function 'my_file.py:true_function' \
#     --metric mse  # Only supported option for now.

"""

from typing import Tuple, Callable
import sys
import os
import argparse

import torch
import numpy as np

from ..estimation import MODELS, MREstimator, MREstimatorWithUncertainty
from . import mse


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="ml-mr evaluation"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to an estimated model results."
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=list(MODELS.keys()),
        help="Name of the estimator so that the results can be loaded "
             "correctly."
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
        default="-3,3",
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

    plt.scatter(
        xs.numpy(),
        true_y.numpy().reshape(-1),
        s=1,
        label="True Y"
    )
    plt.scatter(
        xs.numpy(),
        y_hat.numpy().reshape(-1),
        s=1,
        label="Predicted Y"
    )
    if uncertainty:
        plt.fill_between(
            xs.numpy(),
            y_hat_l.numpy().reshape(-1),
            y_hat_u.numpy().reshape(-1),
            zorder=-1,
            color="#eeeeee"
        )

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args(sys.argv[2:])

    # Load MREstimator.
    estimator = MODELS[args.model]["load"](args.input)

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
    domain_lower, domain_upper = [float(i) for i in args.domain.split(",")]

    cur_mse = mse(
        estimator, true_function, domain=(domain_lower, domain_upper)
    )

    print(cur_mse)

    if args.plot:
        plot(estimator, true_function, domain=(domain_lower, domain_upper))
