"""Example simple sweep for DeepIV.

{
    "sweep": {
        "max_runs": 3,
        "run_output": "my_sweep_run_{run_id}"
    },
    "dataset": {
        "filename": "../ml_mr/test_data/basic_model_data.csv.gz",
        "sep": ",",
        "exposure": "exposure",
        "outcome": "outcome",
        "instruments": ["v1", "v2"]
    },
    "parameters": [
        {
            "name": "exposure_learning_rate",
            "sampler": "grid",
            "start": 1e-4,
            "stop": 0.01,
            "n_values": 3,
            "log": true
        },
        {
            "name": "outcome_learning_rate",
            "sampler": "random_uniform",
            "low": 1e-4,
            "high": 0.01
        },
        {
            "name": "outcome_weight_decay",
            "sampler": "list",
            "values": [0, 1e-2]
        }
    ]
}

# TODO if it's a pure grid, we need to be deterministic in the number of runs.

"""


import os
import sys
import json
import sqlite3
import argparse
import itertools

import numpy as np

from ..estimation.core import IVDataset


SAMPLERS = {}


class sampler_mode:
    def __init__(self, mode_name):
        self.mode_name = mode_name

    def __call__(self, cls):
        global SAMPLERS
        SAMPLERS[self.mode_name] = cls
        return cls


class SweepConfig:
    def __init__(self, dataset, run_output, parameters, max_runs):
        self.dataset = dataset
        self.run_output = run_output
        self.parameters = parameters
        self.max_runs = max_runs

    def print(self):
        print("*** Sweep configuration ***")
        print("[dataset]")
        print(self.dataset)
        print()

        print("[sweep_config]")
        print(f"=> Runs will be saved following template: '{self.run_output}'")
        print(f"=> Max number of runs: '{self.max_runs}'")
        print()

        print("[parameters]")
        for parameter in self.parameters:
            print(parameter)


class Sampler:
    def __iter__(self):
        # Should be an infinite generator.
        raise NotImplementedError()


class SweepParameter:
    def __init__(self, name: str, sampler: Sampler):
        self.name = name
        self.sampler = sampler

    def __repr__(self):
        return f"<Parameter: {self.name} - {self.sampler}>"

    def get_instances(self, n: int = 1):
        for _ in range(n):
            yield from self.sampler


@sampler_mode("grid")
class GridSampler:
    def __init__(self, start, stop, n_values=None, step=None,
                 log: bool = False):
        if n_values is not None and step is not None:
            raise ValueError("Provide step OR n_values.")

        if n_values is not None:
            if log:
                self._values = np.geomspace(start, stop, n_values)
            else:
                self._values = np.linspace(start, stop, n_values)

        elif step is not None:
            if log:
                raise ValueError("Step not implemented in log space.")
            self._values = np.arange(start, stop, step)

        else:
            raise ValueError("Provide either step or n_values.")

    def __iter__(self):
        for v in itertools.cycle(self._values):
            yield v


@sampler_mode("list")
class ListSampler:
    def __init__(self, values: list):
        self.values = values

    def __iter__(self):
        for v in itertools.cycle(self.values):
            yield v


@sampler_mode("random_uniform")
class RandomUniform:
    def __init__(self, low, high, log=False):
        self.log = log

        if log:
            self.buffer_gen = lambda: np.random.uniform(
                np.log(low), np.log(high), size=2
            )
        else:
            self.buffer_gen = lambda: np.random.uniform(
                low, high, size=2
            )

    def __iter__(self):
        while True:
            for element in self.buffer_gen():
                if self.log:
                    yield np.exp(element)
                else:
                    yield element


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="ml-mr sweep"
    )

    parser.add_argument(
        "configuration",
        type=str,
        help="Path to JSON configuration file."
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(os.cpu_count() - 2, 1)
    )

    return parser.parse_args(argv)


def parse_parameter(parameter):
    if "name" not in parameter:
        raise ValueError("Parameter missing a 'name'.")

    if "sampler" not in parameter:
        raise ValueError("Parameter missing a 'sampler'.")

    name = parameter.pop("name")

    # Get the sampler.
    sampler = parameter.pop("sampler")
    sampler_class = SAMPLERS[sampler]

    return SweepParameter(name, sampler_class(**parameter))


def parse_config(filename: str) -> SweepConfig:
    with open(filename, "rt") as f:
        config = json.load(f)

    # Parse the sweep config.
    sweep_conf = config.get("sweep", {})
    max_runs = sweep_conf.get("max_runs", 10_000)  # Default max is 10k runs.
    run_output = sweep_conf.get("run_output", "{run_id}_ml_mr_sweep")

    # Parse the dataset.
    dataset = IVDataset.from_json_configuration(config["dataset"])

    # Parse the parameters.
    if "parameters" not in config:
        raise ValueError(
            "Configuration needs to provide at least one parameter for the "
            "sweep."
        )

    parameters = []
    for parameter in config["parameters"]:
        parameters.append(parse_parameter(parameter))

    return SweepConfig(dataset, run_output, parameters, max_runs)


def populate_run_database(sweep_config: SweepConfig) -> str:
    return "ml_mr_sweep_runs.db"


def execute_runs(sweep_db_filename: str, n_workers: int):
    pass


def main():
    args = parse_args(sys.argv[2:])
    conf = parse_config(args.configuration)

    conf.print()

    database = populate_run_database(conf)

    execute_runs(database, args.n_workers)
