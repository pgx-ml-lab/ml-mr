#!/usr/bin/env python

from itertools import cycle
import argparse
import json
import scipy
import os

import torch
import numpy as np
from ml_mr.estimation import MODELS
from ml_mr.estimation.ensemble import EnsembleMREstimator
from ml_mr.estimation.core import RescaledMREstimator
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimates", nargs="+")
    parser.add_argument("--output", "-o", default="iv_ensembles.png", type=str)
    parser.add_argument("--do-exp", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--color", default=None)
    parser.add_argument(
        "--domain", default=None,
        help="Custom range for plotting in the form of (-2.3, 4.5)"
    )
    parser.add_argument("--domain-95", action="store_true")
    parser.add_argument(
        "--x0", "-x0",
        help="Set the reference comparator for the ATE/CATE.",
        type=float, default=0.0
    )
    parser.add_argument("--effect-fixed-increase", type=float, default=None)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--iv-reg", action="store_true")

    # Display the X-axis in rescaled units.
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)

    # example --given 'genetic_male=0,genetic_male=1'
    parser.add_argument("--given", type=str, default=None)

    args = parser.parse_args()

    if args.effect_unit_increase and args.iv_reg:
        raise ValueError(
            "--iv-reg plots the raw regression function, there is no concept "
            "of reference point."
        )

    return args


def color_generator(num_colors=10, colormap='tab10'):
    cmap = plt.get_cmap(colormap)
    colors = cmap(range(num_colors))
    color_cycle = cycle(colors)
    while True:
        yield next(color_cycle)


def main():
    args = parse_args()

    if args.show_all and args.given is None and args.color is None:
        args.color = "black"

    estimators = []
    covar_labels = None

    for dirname in args.estimates:
        try:
            with open(os.path.join(dirname, "meta.json")) as f:
                meta = json.load(f)
                domain = meta["domain"]
                domain95 = meta.get("exposure_95_percentile")
        except FileNotFoundError:
            print("Can't load estimator in ", dirname)
            continue

        if covar_labels is None:
            covar_labels = meta["covariable_labels"]

        estimators.append(MODELS[meta["model"]]["load"](dirname))

    ensemble = EnsembleMREstimator(*estimators)

    if args.domain:
        low, high = ((
            float(bound) for bound in args.domain.strip("()").split(",")
        ))
        xs = torch.linspace(low, high, 100)
    elif args.domain_95:
        xs = torch.linspace(*domain95, 100)
    else:
        xs = torch.linspace(*domain, 100)

    if args.shift or args.scale:
        shift = args.shift if args.shift else 0
        scale = args.scale if args.scale else 1
        scaler = RescaledMREstimator(ensemble, shift, scale)

        xs = scaler.z_to_x(xs)
        ensemble = scaler

    if args.given is None:
        # Plot ATE.
        if args.iv_reg:
            ate = ensemble.avg_iv_reg_function(
                xs.reshape(-1, 1), ensemble.covars, reduce=False
            )
        else:
            if args.effect_fixed_increase is not None:
                ate = ensemble.ate(
                    xs.reshape(-1, 1),
                    xs.reshape(-1, 1) + args.effect_fixed_increase,
                    reduce=False
                )
            else:
                ate = ensemble.ate(
                    torch.tensor([[args.x0]]), xs.reshape(-1, 1), reduce=False
                )
        plot(xs, ate, args.output, args.show_all, args.do_exp,
             args.iv_reg, args.color, args.alpha, args.effect_fixed_increase)

    else:
        # For all CATE, plot.
        cates = []
        givens = parse_given(args.given)
        if args.color is None:
            color_gen = color_generator(len(givens))

        for given in givens:
            col, val = given
            col_idx = covar_labels.index(col)

            covars = torch.clone(ensemble.covars)
            covars[:, col_idx] = float(val)

            if args.iv_reg:
                cur_cates = ensemble.avg_iv_reg_function(
                    xs.reshape(-1, 1), covars=covars, reduce=False
                )
            else:
                if args.effect_fixed_increase is not None:
                    cur_cates = ensemble.cate(
                        xs.reshape(-1, 1),
                        xs.reshape(-1, 1) + args.effect_fixed_increase,
                        covars=covars, reduce=False
                    )
                else:
                    cur_cates = ensemble.cate(
                        torch.tensor([[args.x0]]), xs.reshape(-1, 1),
                        covars=covars, reduce=False
                    )

            plot(
                xs, cur_cates, args.output, args.show_all, args.do_exp,
                args.iv_reg,
                args.color if args.color is not None else next(color_gen),
                args.alpha, args.effect_fixed_increase, label=f"{col} = {val}"
            )

            if len(givens) >= 2:
                # cate_x by bs
                cates.append(cur_cates)

        if cates and not args.iv_reg:
            # Compare ATEs
            # We do ANOVA of the bagging mean ATE.
            p = scipy.stats.f_oneway(*[
                cates.numpy().reshape(-1) for cates in cates
            ]).pvalue
            print("P:", p)


def parse_given(given):
    return [i.strip().split("=") for i in given.split(",")]


def plot(xs, estimates, output, show_all=False, do_exp=False, iv_reg=False,
         color=None, alpha=0.05, effect_fixed_increase=None, label=None):
    if show_all:
        n_estimators = estimates.shape[1]
        for j in range(n_estimators):
            ys = estimates[:, j].numpy()
            if do_exp:
                ys = np.exp(ys)

            plt.plot(xs.numpy().reshape(-1), ys, lw=0.5, alpha=0.1,
                     color=color)

    agg = torch.quantile(
        estimates, torch.Tensor([alpha / 2, 0.5, 1 - alpha / 2]), dim=1
    ).T.reshape(-1, 1, 3)

    if do_exp:
        agg = torch.exp(agg)

    agg = agg.numpy()

    plt.fill_between(
        xs.numpy().reshape(-1),
        agg[:, 0, 0],
        agg[:, 0, 2],
        alpha=0.1
    )
    plt.plot(
        xs.numpy().reshape(-1),
        agg[:, 0, 1],
        color=color,
        label=label,
        lw=1,
    )

    if label:
        plt.legend()

    if effect_fixed_increase is not None:
        plt.xlabel(r"$X_0$")
    else:
        plt.xlabel("X")

    if iv_reg:
        plt.ylabel("IV Regression function")
    elif effect_fixed_increase is not None:
        plt.ylabel(
            "Average treatment effect (X compared to X+{:.2f})".format(
                effect_fixed_increase
            )
        )
    else:
        plt.ylabel("Average treatment effect (compared to reference value)")

    if not iv_reg:
        plt.axhline(y=1 if do_exp else 0, ls="-.", lw=1, color="#333333")
    if output.endswith(".png"):
        plt.savefig(output, dpi=400)
    else:
        plt.savefig(output)


if __name__ == "__main__":
    main()
