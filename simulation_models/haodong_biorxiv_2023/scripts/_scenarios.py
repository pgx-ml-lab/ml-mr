import torch

import ml_mr.simulation as mr_sim


# Models for the effect of the instrument on the exposure.
# Scenario A: Linear and homogeneous.
@mr_sim.variable
def scenario_a(sim: mr_sim.Simulation):
    return (
        0.5 * sim.get_variable_data("Z") +
        sim.get_variable_data("U") +
        sim.get_variable_data("e_x")
    )


# Scenario B: Non-linear and homogeneous.
@mr_sim.variable
def scenario_b(sim: mr_sim.Simulation):
    indicator = sim.get_variable_data("Z") <= -1

    x = (
        0.5 * sim.get_variable_data("Z") +
        sim.get_variable_data("U") +
        sim.get_variable_data("e_x")
    )

    x[indicator] += 2 * sim.get_variable_data("Z") ** 3 + 2

    return x


# Scenario C: Linear and heterogeneous.
@mr_sim.variable
def scenario_c(sim: mr_sim.Simulation):
    U = sim.get_variable_data("U")
    Z = sim.get_variable_data("Z")
    return (
        -10 +
        (1.5 + 0.4 * U) * Z +
        U +
        sim.get_variable_data("e_x")
    )


# Scenario D: Linear and homogeneous + rounding to nearest integer.
@mr_sim.variable
def scenario_d(sim: mr_sim.Simulation):
    x = scenario_a(sim)
    return round(x)


# Models for the causal relationship.
# Scenario 1: No causal effect of the exposure.
@mr_sim.variable
def scenario_1(sim: mr_sim.Simulation):
    sim.parameters["true_effect"] = 0
    return sim.get_variable_data("U") + sim.get_variable_data("e_y")


def scenario_1_f(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


# Scenario 2: U-shaped causal effect of the exposure on the outcome.
@mr_sim.variable
def scenario_2(sim: mr_sim.Simulation):
    X = sim.get_variable_data("X")
    U = sim.get_variable_data("U")
    return 0.1 * X ** 2 + U + sim.get_variable_data("e_y")


def scenario_2_f(x: torch.Tensor) -> torch.Tensor:
    return 0.1 * x ** 2


# Scenario 3: Threshold causal effect of the exposure on the outcome.
@mr_sim.variable
def scenario_3(sim: mr_sim.Simulation):
    X = sim.get_variable_data("X")
    U = sim.get_variable_data("U")

    indicator = X <= 0

    y = U + sim.get_variable_data("e_y")
    y[indicator] += -0.1 * X ** 2

    return y


def scenario_3_f(x: torch.Tensor) -> torch.Tensor:
    indicator = x > 0
    y = torch.zeros_like(x)
    y[indicator] = -0.1 * x[indicator] ** 2
    return y


INSTRUMENT_EXPOSURE_SCENARIOS = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
    "D": scenario_d,
}


EXPOSURE_OUTCOME_SCENARIOS = {
    1: scenario_1,
    2: scenario_2,
    3: scenario_3,
}


TRUE_CAUSAL_FUNCTIONS = {
    1: scenario_1_f,
    2: scenario_2_f,
    3: scenario_3_f
}
