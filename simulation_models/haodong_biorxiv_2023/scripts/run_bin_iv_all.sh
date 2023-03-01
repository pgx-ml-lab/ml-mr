#!/usr/bin/env bash

export SHELL=$(type -p bash)

dobiniv() {
    scenario=$(echo $1 | grep -E -o 'haodong-scenario.+$' | cut -f 1 -d '_')
    output="../results/${scenario}_bin_iv"

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
        --wandb-project $scenario
}

export -f dobiniv

parallel --jobs 1 dobiniv ::: ../simulated_datasets/haodong-scenario-*.csv.gz
