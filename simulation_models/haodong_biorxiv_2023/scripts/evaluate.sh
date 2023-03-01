#!/usr/bin/env bash


# Evaluate all the models in results.

echo "instrument_exposure_scenario,exposure_outcome_scenario,mse"

doeval() {
    filename=$1

    # Find the scenario number.
    scenario=$(echo $filename | grep -Eo '\-([A-D][1-3])_bin_iv' | sed 's/-//' | sed 's/_bin_iv//')

    echo -n "${scenario:0:1},${scenario:1:1},"

    ml-mr evaluation \
        --input $filename \
        --model bin_iv \
        --true-function _scenarios.py:scenario_${scenario:1:1}_f \

}

export -f doeval

find ../results -name '*_bin_iv' -exec bash -c 'doeval "$0"' {} \;
