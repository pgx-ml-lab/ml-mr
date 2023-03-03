#!/usr/bin/env bash

export SHELL=$(type -p bash)

dobiniv() {
    scenario=$(echo $1 | grep -E -o 'tian-scenario.+$' | cut -f 1 -d '_')
    output="../results/${scenario}_bin_iv"

    run_name=$(echo $scenario | sed 's/tian-scenario-//')

    ml-mr estimation bin_iv \
        --n-bins 15 \
        --output-dir $output \
        --sqr \
        --outcome-type continuous \
        --data $1 \
        --sep ',' \
        --instruments Z \
        --exposure X \
        --outcome Y \
        --wandb-project "tian_2022_sim:${run_name}_deep_iv"
}

export -f dobiniv

parallel --jobs 1 dobiniv ::: ../simulated_datasets/tian-scenario-*.csv.gz
