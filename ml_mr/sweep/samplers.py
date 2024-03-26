import itertools

import numpy as np


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


@sampler_mode("random_uniform_int")
class RandomUniformInt(StochasticSampler):
    def __init__(self, low, high):
        super().__init__("integer")
        self.buffer_gen = lambda: np.random.randint(low, high + 1, size=1024)

    def __iter__(self):
        while True:
            for element in self.buffer_gen():
                yield element.item()


@sampler_mode("random_uniform")
class RandomUniform(StochasticSampler):
    def __init__(self, low, high, log=False):
        super().__init__("float")

        self.log = log

        if log:
            self.buffer_gen = lambda: np.random.uniform(
                np.log(low), np.log(high), size=1024
            )
        else:
            self.buffer_gen = lambda: np.random.uniform(
                low, high, size=1024
            )

    def __iter__(self):
        while True:
            for element in self.buffer_gen():
                if self.log:
                    yield np.exp(element)
                else:
                    yield element
