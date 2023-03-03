#!/usr/bin/env python

import os
import itertools

from _scenarios import (EXPOSURE_OUTCOME_SCENARIOS,
                        INSTRUMENT_EXPOSURE_SCENARIOS)

import ml_mr.simulation as mr_sim


def main():
    # In the original paper, they use N = 10,000
    N = 100_000

    for i_e_label, e_o_label in itertools.product(
        INSTRUMENT_EXPOSURE_SCENARIOS,
        EXPOSURE_OUTCOME_SCENARIOS
    ):
        # Instrument-exposure model.
        i_e_variable = INSTRUMENT_EXPOSURE_SCENARIOS[i_e_label]

        # Exposure-outcome model.
        e_o_variable = EXPOSURE_OUTCOME_SCENARIOS[e_o_label]

        sim = mr_sim.Simulation(
            N,
            prefix=os.path.join(
                "..",
                "simulated_datasets",
                f"tian-scenario-{i_e_label}{e_o_label}"
            )
        )

        variables = [
            mr_sim.Normal("U", 0, 1),
            mr_sim.Normal("Z", 0, 0.5),
            mr_sim.Normal("e_x", 0, 1),
            mr_sim.Normal("e_y", 0, 1),
        ]
        sim.add_variables(variables)

        # Create the exposure model.
        exposure = i_e_variable
        exposure.name = "X"
        sim.add_variable(exposure)

        outcome = e_o_variable
        outcome.name = "Y"
        sim.add_variable(outcome)

        sim.save()


if __name__ == "__main__":
    main()
