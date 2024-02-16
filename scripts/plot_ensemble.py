#!/usr/bin/env python

import argparse
import json
import os

import torch
import numpy as np
from ml_mr.estimation import MODELS
from ml_mr.estimation.core import EnsembleMREstimator
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimates", nargs="+")
    parser.add_argument("--output", "-o", default="iv_ensembles.png", type=str)
    parser.add_argument("--do-exp", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--color", default="black")
    parser.add_argument("--domain-95", action="store_true")
    parser.add_argument("--alpha", default=0.05, type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    estimators = []

    for dirname in args.estimates:
        try:
            with open(os.path.join(dirname, "meta.json")) as f:
                meta = json.load(f)
                domain = meta["domain"]
                domain95 = meta.get("exposure_95_percentile")
        except FileNotFoundError:
            print("Can't load estimator in ", dirname)
            continue

        estimators.append(MODELS[meta["model"]]["load"](dirname))

    ensemble = EnsembleMREstimator(*estimators)

    if args.domain_95:
        xs = torch.linspace(*domain95, 100)
    else:
        xs = torch.linspace(*domain, 100)

    ate = ensemble.ate(torch.tensor([[0.0]]), xs.reshape(-1, 1), reduce=False)

    if args.show_all:
        n_estimators = ate.shape[1]
        for j in range(n_estimators):
            ys = ate[:, j].numpy()
            if args.do_exp:
                ys = np.exp(ys)

            plt.plot(xs.numpy().reshape(-1), ys, lw=0.5, alpha=0.1,
                     color=args.color)

    ate = torch.quantile(
        ate, torch.Tensor([args.alpha / 2, 0.5, 1 - args.alpha / 2]), dim=1
    ).T.reshape(-1, 1, 3)

    if args.do_exp:
        ate = torch.exp(ate)

    ate = ate.numpy()

    plt.fill_between(
        xs.numpy().reshape(-1),
        ate[:, 0, 0],
        ate[:, 0, 2],
        alpha=0.1
    )
    plt.plot(
        xs.numpy().reshape(-1),
        ate[:, 0, 1],
        color=args.color
    )

    plt.xlabel("X")
    plt.ylabel("Average treatment effect (compared to X=0)")

    plt.axhline(y=1 if args.do_exp else 0, ls="-.", lw=1, color="#333333")
    if args.output.endswith(".png"):
        plt.savefig(args.output, dpi=400)
    else:
        plt.savefig(args.output)


if __name__ == "__main__":
    main()
