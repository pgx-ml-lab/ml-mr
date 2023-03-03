#!/usr/bin/env bash

export SHELL=$(type -p bash)

doquantileiv() {
    scenario=$(echo $1 | grep -E -o 'haodong-scenario.+$' | cut -f 1 -d '_')
    output="../results/${scenario}_quantile_iv"

    run_name=$(echo $scenario | sed 's/haodong-scenario-//')

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
        --wandb-project "haodong_2023_sim:${run_name}_quantile_iv"
}

export -f doquantileiv

parallel --jobs 1 doquantileiv ::: ../simulated_datasets/haodong-scenario-*.csv.gz

