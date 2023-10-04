"""
Utilities to create simulated datasets to evaluate Mendelian randomization
methods.

What kind of API do we want?

sim = Simulation(n=100)

gs = [
    Variant(name=f"v{i}", frequency=np.random.rand())
    for i in range(10)
]
sim.add_variables(gs)

u = lambda sim: np.random.random(sim.n)  # Can be callable.
m = np.random.random(100)  # Or vector directly.
sim.add_variables({"u": u, "m": m})
# or sim.add_variable("u", u)

# Will get recorded in JSON and can be retrieved in callable simulation.
sim.add_parameter("h2", 0.4)

def y(sim):
    genetic_effects = (
        np.random.randn(100) + sim.get("m") * np.random.randn(100)
    )
    x = sim.get_values("x")
    return x @ genetic_effects + np.random.random(sim.n)

"""

from collections import OrderedDict
from typing import Optional, Dict, Any, Iterable

import json
import pandas as pd
import numpy as np

from .. import logging


class Simulation:
    def __init__(self, n: int, prefix: str = "mr_simulation"):
        self.n = n
        self.prefix = prefix

        # Individual-level simulated data.
        self._data = pd.DataFrame(index=range(n))

        # Simulation parameters. They can be fixed or stochastic but they are
        # represented as Variable subclasses.
        self._sim_parameters: Dict[str, Variable] = OrderedDict()
        self._sim_parameters_values: Dict[str, Any] = OrderedDict()

        # Variables can be stochastic or fixed. Their realization are stored
        # in _data.
        self._sim_variables: Dict[str, Variable] = {}

    @property
    def parameters(self):
        class _ParameterDict:
            def __getitem__(self2, name: str) -> Any:
                return self.get_sim_parameter(name)

            def __setitem__(self2, name: str, value: Any) -> None:
                self.add_sim_parameter(name, value)

        return _ParameterDict()

    def _check_var_name(self, name, strict=False):
        if name in self._sim_variables:
            message = f"Variable named '{name}' already exists"
            if strict:
                raise ValueError(message)
            else:
                logging.warn(message + " (overwriting).")

    def add_variables(self, variables: Iterable["Variable"]):
        variables_li = list(variables)
        for v in variables_li:
            self._check_var_name(v.name)
            self._sim_variables[v.name] = v

        mat = np.hstack([v(self).reshape(-1, 1) for v in variables_li])
        df = pd.DataFrame(mat, columns=[v.name for v in variables_li])
        self._data = pd.concat((self._data, df), axis=1)

    def add_variable(self, *args):
        """Add a variable to the simulation model.

        Args can be either:

            - name, data
            - a Variable instance

        """
        if len(args) == 1:
            variable = args[0]
            self._check_var_name(variable.name)
            self._sim_variables[variable.name] = variable
            self._data[variable.name] = variable(self)

        elif len(args) == 2:
            name, data = args
            variable = Variable(name, data)
            return self.add_variable(variable)

    def add_sim_parameter(self, *args):
        if len(args) == 1:
            variable = args[0]
            self._sim_parameters[variable.name] = variable
            self._sim_parameters_values[variable.name] = variable(self)

        elif len(args) == 2:
            name, data = args
            variable = Variable(name, data)
            return self.add_sim_parameter(variable)

    def get_sim_parameter(self, name):
        return self._sim_parameters_values[name]

    def get_variable(self, name):
        return self._sim_variables[name]

    def get_variable_data(self, name):
        return self._data[name].values

    def resample(self):
        for name, variable in self._sim_parameters.items():
            self._sim_parameters_values[name] = variable(self)

        self._data = pd.DataFrame(index=range(self.n))
        for name, variable in self._sim_variables.items():
            self._data[name] = variable(self)

    def get_parameters_dict(self) -> Dict[str, Any]:
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self._sim_parameters_values.items()
        }

    def save(self, index: Optional[int] = None):
        # Save JSON with sim parameters and pandas dataframe as csv.
        base = self.prefix
        if index is not None:
            base += f"_{index}"

        with open(f"{base}_sim_parameters.json", "wt") as f:
            json.dump(self.get_parameters_dict(), f, indent=4)

        self._data.to_csv(
            f"{base}_sim_data.csv.gz",
            compression="gzip",
            index=False
        )

    def save_pickle(self, filename):
        try:
            import dill as pickle
        except ImportError:
            raise RuntimeError(
                "The 'dill' package is needed to serialize simulations.\n"
                "Use 'pip install dill' to install it."
            )

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename):
        import dill as pickle
        with open(filename, "rb") as f:
            return pickle.load(f)


class Variable:
    def __init__(self, name, data=None):
        self.name = name
        if data is not None:
            self._data = data

    def __call__(self, sim: Simulation):
        if self._data is not None:
            return self._data
        else:
            raise ValueError(f"No bound data for variable '{self.name}'.")


class Variant(Variable):
    def __init__(self, name, frequency):
        super().__init__(name)
        self.frequency = frequency

    def __call__(self, sim: Simulation):
        return np.random.binomial(2, self.frequency, size=sim.n)


class Normal(Variable):
    def __init__(
        self,
        name: str,
        mu: float,
        sigma: float,
        size: Optional[int] = None
    ):
        super().__init__(name)
        self.mu = mu
        self.sigma = sigma
        self.size = size

    def __call__(self, sim: Simulation):
        if self.size is None:
            size = sim.n
        else:
            size = self.size

        return np.random.normal(self.mu, self.sigma, size=size)


class Uniform(Variable):
    def __init__(
        self,
        name: str,
        low: float = 0.0,
        high: float = 1.0,
        size: Optional[int] = None
    ):
        super().__init__(name)
        self.low = low
        self.high = high
        self.size = size

    def __call__(self, sim: Simulation):
        if self.size is None:
            size = sim.n
        else:
            size = self.size

        return np.random.uniform(low=self.low, high=self.high, size=size)
