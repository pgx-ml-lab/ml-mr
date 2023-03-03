#!/usr/bin/env bash

export SHELL=$(type -p bash)

doquantileiv() {
    scenario=$(echo $1 | grep -E -o 'tian-scenario.+$' | cut -f 1 -d '_')
    output="../results/${scenario}_quantile_iv"

    run_name=$(echo $scenario | sed 's/tian-scenario-//')

    ml-mr estimation quantile_iv \
        --q 8 \
        --output-dir $output \
        --sqr \
        --outcome-type continuous \
        --data $1 \
        --sep ',' \
        --instruments Z \
        --exposure X \
        --outcome Y \
        --exposure-add-input-batchnorm \
        --outcome-add-input-batchnorm \
        --wandb-project "tian_2022_sim:${run_name}_quantile_iv"
}

export -f doquantileiv

parallel --jobs 1 doquantileiv ::: ../simulated_datasets/tian-scenario-*.csv.gz

