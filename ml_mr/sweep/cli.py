import os
import sys
import json
import math
import time
import shutil
import sqlite3
import argparse
import random
import itertools
import multiprocessing
from typing import List, Iterator, Union

import numpy as np

from ..estimation import MODELS
from ..estimation.core import IVDataset
from ..logging import debug, warn, info


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

        db_type = "float"
        if n_values is not None:
            if log:
                self._values = np.geomspace(start, stop, n_values).tolist()
            else:
                self._values = np.linspace(start, stop, n_values).tolist()

        elif step is not None:
            if log:
                raise ValueError("Step not implemented in log space.")
            self._values = np.arange(start, stop, step).tolist()

            if (
                isinstance(start, int) and
                isinstance(stop, int) and
                isinstance(step, int)
            ):
                db_type = "integer"

        else:
            raise ValueError("Provide either step or n_values.")

        super().__init__(db_type, len(self._values))

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
        elif any([isinstance(e, bool) for e in values]):
            db_type = "boolean"
        elif all([isinstance(e, str) for e in values]):
            db_type = "text"
        else:
            raise ValueError(f"Couldn't infer db type from list '{values}'")

        super().__init__(db_type, len(values))

        self.values = values

    def __iter__(self):
        for v in itertools.cycle(self.values):
            yield v


@sampler_mode("literal")
class LiteralSampler(ListSampler):
    def __init__(self, value):
        if type(value) not in {int, float, str, bool}:
            raise ValueError(
                "Literal sampler only supported for int, float, bool and str."
            )
        super().__init__([value])


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
        dataset_config: dict,
        model: str,
        sweep_directory: str,
        parameters: List["SweepParameter"],
        max_runs: int
    ):
        self.dataset_config = dataset_config
        self.model = model
        self.sweep_directory = os.path.abspath(sweep_directory)
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
        print(self.dataset_config)
        print()

        print("[sweep_config]")
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
        help="Path to JSON configuration file or a ml-mr sweep database in "
             "which case the failed runs will be attempted again."
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(os.cpu_count() - 2, 1)
    )

    parser.add_argument(
        "--create-db-only",
        action="store_true"
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
    sweep_directory = os.path.abspath(
        sweep_conf.get("sweep_directory", "ml_mr_sweep")
    )
    model = sweep_conf["model"]

    if model not in MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Accepted values: {list(MODELS.keys())}"
        )

    # Parse the parameters.
    if "parameters" not in config:
        raise ValueError(
            "Configuration needs to provide at least one parameter for the "
            "sweep."
        )

    parameters = []
    for parameter in config["parameters"]:
        parameters.append(parse_parameter(parameter))

    return SweepConfig(
        config["dataset"], model, sweep_directory, parameters, max_runs
    )


def create_sweep_database(sweep_config: SweepConfig) -> str:
    filename = os.path.abspath(os.path.join(
        sweep_config.sweep_directory,
        "ml_mr_sweep_runs.db"
    ))

    con = sqlite3.connect(filename)
    cur = con.cursor()

    # Create the table containing the dataset information.
    cur.execute("create table dataset (json_conf text);")
    cur.execute(
        "insert into dataset values (?)",
        (json.dumps(sweep_config.dataset_config), )
    )
    con.commit()

    # Create the table with the sweep metadata.
    cur.execute(
        "create table meta ("
        "  model text,"
        "  sweep_directory text"
        ");"
    )
    cur.execute(
        "insert into meta values (?, ?)",
        (sweep_config.model, sweep_config.sweep_directory)
    )
    con.commit()

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
        "  in_progress boolean default false,\n"
        "  elapsed float,\n"
        "  failed boolean default false"
        ");"
    )

    # Populate with runs.
    if sweep_config.stochastic:
        debug("Generating parameter table for stochastic sweep.")

        param_samples = []
        for param in sweep_config.parameters:
            samples = list(param.get_instances(sweep_config.max_runs))

            # Ensure deterministic samplers also have random order.
            random.shuffle(samples)

            param_samples.append(samples)

        parameter_table: Iterator = zip(
            itertools.count(0),
            *param_samples
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

        def take(iterator, n_elements):
            for i, element in enumerate(iterator):
                if i >= n_elements:
                    break
                yield element

        parameter_table = take((
            (i,) + params for i, params in enumerate(itertools.product(*[
                param.get_instances(param.sampler.n_elements)  # type: ignore
                for param in sweep_config.parameters
            ]))
        ), sweep_config.max_runs)

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
        "  select run_id, false, false, NULL, false "
        "  from run_parameters;"
    )
    con.commit()
    con.close()

    return filename


def _fetchone_as_dict(cur: sqlite3.Cursor) -> dict:
    return {k[0]: v for k, v in zip(cur.description, cur.fetchone())}


def worker(
    db_filename: str,
    db_lock: multiprocessing.synchronize.Lock,
    stop_flag
):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    os.environ["ML_MR_QUIET"] = "1"

    # We only allow offline mode for wandb otherwise the directory isolation
    # seems to cause problems.
    os.environ["WANDB_MODE"] = "offline"

    # Get the config.
    db_lock.acquire()
    try:
        cur.execute("select * from meta;")
        meta = _fetchone_as_dict(cur)

        cur.execute("select json_conf from dataset;")
        dataset_conf = json.loads(cur.fetchone()[0])
    finally:
        db_lock.release()

    dataset = IVDataset.from_json_configuration(dataset_conf)
    fit_func = MODELS[meta["model"]]["estimate"]

    while True:
        if stop_flag.value:
            debug("Process exiting due to stop flag.")
            return

        db_lock.acquire()
        try:
            # Get a task from the DB.
            cur.execute(
                "select run_id from run_status "
                "where (not done) and (not in_progress) "
                "limit 1;"
            )

            run_id_tu = cur.fetchone()
            if run_id_tu is None:
                # No more pendings tasks.
                debug("Process exiting, no more runs pending.")
                con.close()
                return

            run_id = run_id_tu[0]

            cur.execute(
                "update run_status "
                "set in_progress=true "
                "where run_id=?", (run_id, )
            )
            con.commit()

            cur.execute(
                "select * from run_parameters where run_id=?",
                (run_id, )
            )

            task = _fetchone_as_dict(cur)
            del task["run_id"]

        finally:
            db_lock.release()

        # Do the work (in a subdirectory for isolation).
        root_dir = os.getcwd()
        dir_name = os.path.join(meta["sweep_directory"], str(run_id))
        os.makedirs(dir_name)
        os.chdir(dir_name)
        failed = "false"
        delta_t: Union[str, float] = "null"
        try:
            t0 = time.time()
            fit_func(  # type: ignore
                dataset=dataset,
                output_dir=f"estimate_run_{run_id}",
                accelerator="cpu",
                **task
            )
            t1 = time.time()
            delta_t = t1 - t0
        except Exception as e:  # noqa: E722
            print(e)
            failed = "true"

        finally:
            os.chdir(root_dir)

        # Mark task as done.
        db_lock.acquire()
        try:
            cur.execute(
                f"update run_status "
                f"  set "
                f"    in_progress=false, "
                f"    done=true, "
                f"    elapsed={delta_t}, "
                f"    failed={failed} "
                f"where run_id=?", (run_id, )
            )
            con.commit()
        finally:
            db_lock.release()


def execute_runs(sweep_db_filename: str, n_workers: int):
    proc_ctx = multiprocessing.get_context("spawn")
    processes = []

    db_lock = proc_ctx.Lock()
    stop_flag = proc_ctx.Value("B", 0)

    for _ in range(n_workers):
        proc = proc_ctx.Process(
            target=worker,
            args=[sweep_db_filename, db_lock, stop_flag]
        )
        proc.start()
        processes.append(proc)

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        debug("Catching interrupt. Closing processes.")
        stop_flag.value = 1  # type: ignore

        # Try joining a second time after requesting shutdown.
        for proc in processes:
            proc.join()
    finally:
        # If there are tasks still marked as in_progress after all workers have
        # died, we mark them as failed.
        con = sqlite3.connect(sweep_db_filename)
        cur = con.cursor()

        cur.execute(
            "update run_status "
            "set failed=true, in_progress=false where in_progress=true;"
        )
        con.commit()
        con.close()

    debug("Done all!")


def resume_sweep(db_filename: str, n_workers: int):
    info("Resuming sweep from database.")

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    cur.execute("select * from meta;")
    meta = _fetchone_as_dict(cur)

    # Get the failed runs to cleanup.
    cur.execute("select run_id from run_status where failed=true;")
    failed_run_ids = [tu[0] for tu in cur.fetchall()]

    for run_id in failed_run_ids:
        dir_name = os.path.join(meta["sweep_directory"], str(run_id))
        debug(f"Cleaning up failed run '{dir_name}'.")
        shutil.rmtree(dir_name)

    cur.execute(
        "update run_status "
        "set failed=false, done=false where failed=true;"
    )
    con.commit()
    con.close()

    return execute_runs(db_filename, n_workers)


def main():
    args = parse_args(sys.argv[2:])

    # Check if args.configuration is a database.
    with open(args.configuration, "rb") as f:
        if f.read(16) == b"SQLite format 3\x00":
            if args.create_db_only:
                raise ValueError(
                    "Database provided by ml-mr sweep called with "
                    "--create-db-only."
                )

            return resume_sweep(args.configuration, args.n_workers)

    # Create and run sweep from configuration.
    conf = parse_config(args.configuration)
    conf.print()
    database = create_sweep_database(conf)

    if args.create_db_only:
        return

    execute_runs(database, args.n_workers)
