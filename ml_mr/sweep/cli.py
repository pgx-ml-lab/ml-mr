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

"""


import os
import sys
import json
import math
import sqlite3
import argparse
import itertools
from typing import List, Iterator

import numpy as np

from ..estimation.core import IVDataset
from ..logging import debug, warn


SAMPLERS = {}


class sampler_mode:
    def __init__(self, mode_name):
        self.mode_name = mode_name

    def __call__(self, cls):
        global SAMPLERS
        SAMPLERS[self.mode_name] = cls
        return cls


class Sampler:
    def __init__(self, db_type: str):
        self.db_type = db_type

    def __iter__(self):
        # Should be an infinite generator.
        raise NotImplementedError()


class StochasticSampler(Sampler):
    """Class to denote stochastic samplers.

    The main implication is that if there is at least one parameter with
    a stochastic sampler, runs will be sampled up to the max_runs parameter.

    """
    pass


class DeterministicSampler(Sampler):
    def __init__(self, db_type: str, n_elements: int):
        self.n_elements = n_elements
        super().__init__(db_type)


class SweepParameter:
    def __init__(self, name: str, sampler: Sampler):
        self.name = name
        self.sampler = sampler

    def __repr__(self):
        return f"<Parameter: {self.name} - {self.sampler}>"

    def get_instances(self, n: int = 1):
        iterator = iter(self.sampler)
        for i in range(n):
            yield next(iterator)
            if i >= n:
                break


@sampler_mode("grid")
class GridSampler(DeterministicSampler):
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

        super().__init__("float", len(self._values))

    def __iter__(self):
        for v in itertools.cycle(self._values):
            yield v


@sampler_mode("list")
class ListSampler(DeterministicSampler):
    def __init__(self, values: list):
        # Infer type from list.
        if all([isinstance(e, int) for e in values]):
            db_type = "integer"
        elif any([isinstance(e, float) for e in values]):
            db_type = "float"
        elif all([isinstance(e, str) for e in values]):
            db_type = "text"
        else:
            raise ValueError(f"Couldn't infer db type from list '{values}'")

        super().__init__(db_type, len(values))

        self.values = values

    def __iter__(self):
        for v in itertools.cycle(self.values):
            yield v


@sampler_mode("random_uniform")
class RandomUniform(StochasticSampler):
    def __init__(self, low, high, log=False):
        super().__init__("float")

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


class SweepConfig:
    def __init__(
        self,
        dataset: IVDataset,
        run_output: str,
        parameters: List["SweepParameter"],
        max_runs: int
    ):
        self.dataset = dataset
        self.run_output = run_output
        self.parameters = parameters
        self.max_runs = max_runs

        # Check if at least one parameter has a stochastic sampler.
        self.stochastic = False
        for p in self.parameters:
            if isinstance(p.sampler, StochasticSampler):
                self.stochastic = True
                break

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


def create_sweep_database(sweep_config: SweepConfig) -> str:
    filename = "ml_mr_sweep_runs.db"
    con = sqlite3.connect(filename)
    cur = con.cursor()

    # Create the parameters table.
    create_params = (
        "create table run_parameters (\n"
        "  run_id integer primary key,\n"
    )

    for parameter in sweep_config.parameters:
        create_params += (
            "  `{}` {},\n".format(parameter.name, parameter.sampler.db_type)
        )

    create_params = create_params.strip().rstrip(",") + "\n);"
    debug(create_params)

    cur.execute(create_params)

    # Create the run status table.
    cur.execute(
        "create table run_status (\n"
        "  run_id integer primary key,\n"
        "  done boolean default false,\n"
        "  in_progress boolean default false\n"
        ");"
    )

    # Populate with runs.
    if sweep_config.stochastic:
        debug("Generating parameter table for stochastic sweep.")

        # Sample up to max runs.
        parameter_table: Iterator = zip(
            itertools.count(0),
            *[param.get_instances(sweep_config.max_runs)
              for param in sweep_config.parameters]
        )

        parameter_table = itertools.islice(
            parameter_table, sweep_config.max_runs
        )

    else:
        debug("Generating parameter table for deterministic sweep.")
        n_elements = []
        for parameter in sweep_config.parameters:
            assert isinstance(parameter.sampler, DeterministicSampler)
            n_elements.append(parameter.sampler.n_elements)

        expected_n_runs = math.prod(n_elements)

        if expected_n_runs > sweep_config.max_runs:
            warn(
                f"Specified parameter values have {expected_n_runs} "
                f"parameters, but the specified max_runs is "
                f"{sweep_config.max_runs}. Some parameter combinations will "
                f"not be included in the sweep. Increase the max_runs in the "
                f"sweep config to avoid this."
            )

            n_runs = min(expected_n_runs, sweep_config.max_runs)

        else:
            n_runs = expected_n_runs

        parameter_table = zip(
            itertools.count(0),
            *[param.get_instances(n_runs)
                for param in sweep_config.parameters]
        )

    n_params = len(sweep_config.parameters)
    # Note we have an extra parameter for the run_id.
    val_placeholder = "({})".format("?," * (n_params) + "?")
    cur.executemany(
        f"insert into run_parameters values {val_placeholder}",
        parameter_table
    )

    con.commit()

    # Create the entries in run status.
    cur.execute(
        "insert into run_status "
        "  select run_id, false, false "
        "  from run_parameters;"
    )
    con.commit()

    return "ml_mr_sweep_runs.db"


def execute_runs(sweep_db_filename: str, n_workers: int):
    pass


def main():
    args = parse_args(sys.argv[2:])
    conf = parse_config(args.configuration)

    conf.print()

    database = create_sweep_database(conf)

    execute_runs(database, args.n_workers)
