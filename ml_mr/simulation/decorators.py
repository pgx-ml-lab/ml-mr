from typing import Callable

import numpy as np

from .simulation import Variable, Simulation


class _FunctionalVariable(Variable):
    def __init__(self, f: Callable[[Simulation], np.ndarray]):
        self.name = f.__name__
        self._f = f

    def __call__(self, sim: Simulation):
        return self._f(sim)


def variable(f):
    """Decorator to convert a regular function to a simulation variable."""
    return _FunctionalVariable(f)
