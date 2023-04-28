"""
Simulate the J shaped model (f_3) from the Burgess nonlinear MR paper.
"""

import numpy as np

import ml_mr.simulation as mr_sim


N = 100_000


def f_struct(x):
    return 0.2 * (x - 1) ** 2


sim = mr_sim.Simulation(N, "burgess_epidemiology_J_shaped")

sim.parameters["coded_freq"] = 0.3

sim.add_variable(mr_sim.Variant("g", sim.parameters["coded_freq"]))
sim.add_variable(mr_sim.Uniform("u"))


@mr_sim.variable
def exposure(sim: mr_sim.Simulation):
    return (
        0.25 * sim.get_variable_data("g") +
        sim.get_variable_data("u") +
        np.random.exponential(scale=1, size=N)
    )


@mr_sim.variable
def outcome(sim: mr_sim.Simulation):
    return (
        f_struct(sim.get_variable_data("exposure")) +
        0.8 * sim.get_variable_data("u") +
        np.random.normal(size=N)
    )


sim.add_variable(exposure)
sim.add_variable(outcome)
