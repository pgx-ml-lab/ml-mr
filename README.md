<img src="assets/ml_mr_logo_web.png?raw=true" alt="ML-MR Logo" width="200" />

_Machine learning instrumental variable estimators for Mendelian randomization_

# Introduction

The goal of _ml-mr_ is to provide a toolkit to facilitate the use of machine learning estimators developed for instrumental variable causal effect estimation in Mendelian randomization studies.

There are 4 core modules:

1. **Simulation** Utilities to simulate data and pre-implemented simulation models from published articles.
2. **Estimation** Unified interface (programmatic and command-line) to different estimators. This module makes it easy to compare algorithms, and to train the neural networks on arbitrary hardware (_e.g._ GPUs) while not having to worry about logging, early stopping, model checkpointing, etc.
3. **Evaluation** Facilitates plotting, computing metrics when the true causal function is known (_e.g._ in simulation models). It also makes it easy to extract stored model statistics or metadata.
4. **Sweep** Many of the implemented algorithms rely on hyperparameters. For example, we need to pre-specify the number of quantiles to use in Quantile IV, or the learned feature's dimensionality in the Deep Feature IV algorithm. Furthermore, neural network optimization also relies on hyperparameters such as the learning rate. To make it easy to conduct grid or random hyperparameter searches, we implemented a sweep module that uses multiprocessing to fit models in parallel.

# Estimands

To help understand the different estimands available in _ml-mr_, we give a short clarifying description. Throughout the table, $X$ refers to the exposure, $Y$ the outcome and $C$ observed covariables.

| [MREstimator](ml_mr/estimation/core.py) method or standard terminology | Mathematical object | Definition | Comments |
| --- | --- | --- | --- |
| `iv_reg_function` | $f: X,C \to Y$ | $\mathbb{E}[Y \vert \text{do}(X),C] $ | Estimates the structural function between the exposure and the outcome conditional on specified covariate values. It is often the case that this estimate will biased. |
`avg_iv_reg_function` | $f: X \to Y$ | $\mathbb{E}[Y \vert \text{do}(X)]$ | In practice, we marginalize the covariates from the __iv_reg_function__ by using the observed covariates values. In other words, we use the sample estimate of $\mathbb{E}[Y \vert \text{do}(X)] = \int \mathbb{E}[Y \vert \text{do}(X), C] dF_c$ which we take as: $$\frac{1}{n}\sum_{i=1}^n \mathbb{E}[Y \vert \text{do}(X),C=c_i].$$ |
| `ate` (Average treatment effect) | Scalar | $\mu_1 - \mu_0$ with $\mu_x = \mathbb{E}[Y \vert \text{do}(X=x)]$ | This can be expressed as the difference of __avg_iv_reg_function__ calls at reference values (_e.g._ $X_1=1$ and $X_0=0$). | 
| `cate` (Conditional Average Treatment Effect) | Scalar at fixed value of C or $f: C \to \mathbb{R}$ | $\mu_{1c} - \mu_{0c}$ with $\mu_{xc} = \mathbb{E}[Y \vert \text{do}(X=x), C=c]$ | This quantity is also defined with respect to a pair of X values that are compared and fixed levels of the observed covariates. |

# GLBIO Presentation

We presented _ml-mr_ at the [Great Lakes Bioinformatics 2023 conference](https://www.iscb.org/glbio2023). The [slides are available here](assets/ml_mr_GLBIO.pdf).

# Installation

```
# Clone repository.
git clone git@github.com:legaultmarc/ml-mr.git
pip install ml-mr
```

We recommend setting up [Weights and Biases](https://wandb.ai/quickstart). It is integrated with our estimator module and will help track training metrics to ensure convergence and diagnose problems.

# Simulation

A simple simulation model from Burgess S, _et al._ _Epidemiology_ (2014) can be found [here](simulation_models/burgess_epidemiology_2014/scripts/simulate.py).

A more complex example can be found [here](https://gist.github.com/legaultmarc/6083df23150c463879e32e6184459ca9). In this model, the (linear) causal effect of the exposure on the outcome varies nonlinearly as a function of an effect modifier variable. It is a good example of where machine learning models may perform better than conventional instrumental variable estimators.

# Estimation

Assuming a TSV file with columns for the instruments, covariables, exposure and outcome, we can easily fit machine learning instrumental variable estimators using the command-line interface.

Here's an example command that would train a _Quantile IV_ model.

```bash
ml-mr estimation \
  quantile_iv \
  --n-quantiles 10 \
  --data filename.tsv.gz \
  --exposure x --outcome y --instruments v{1..20} \
  --wandb-project my_fit
```

## Implemented algorithms

| Algorithm | Reference | Comments |
| --------- | --------- | -------- |
| Deep IV   | Hartford J, _et al._ _ICLR_ (2017) | We implement three different density estimators for the 1st stage: ridge regression, mixture density network (as proposed in the original paper) and a gaussian network. We recommend using the **gaussian network** which performed best in simulation studies. This can be done using the ``--exposure-network-type gaussian_net`` option. |
| Quantile IV | Legault MA, _et al._ medRxiv (2024) | We derived a new estimator we call Quantile IV. It is similar to Deep IV but uses quantile regression to learn quantiles of the conditional exposure distribution. Instead of sampling, it then uses quantile midpoints to train the outcome network allowing for simple averaging. |
| Deep feature IV | Xu L, _et al._ _ICLR_ (2021) | This method also has two stages, but it is quite different from Deep IV and others. It learns feature mappings for the instruments and for the exposure and estimates the causal effect using penalized two stage least squares regression on these learned features. |

In the future, we hope to include algorithms that rely on deep generalized method of moments and on kernel instrumental variable methods. Other baseline methods (_e.g._ LACE estimator, DeLIVR, 2SLS) are implemented, but they are meant to be used for simple benchmarking and comparisons.


## Example of the command-line utility documentation / help page

The command line interface integrated help message details the full array of option for every estimator. For example, the help message from _Deep IV_ is shown below:

<details>
<summary>Display command-line help page</summary>

```
usage: ml-mr estimation deep_iv [-h] [--n-gaussians N_GAUSSIANS] [--exposure-network-type {mixture_density_net,gaussian_net,ridge}]
                                [--output-dir OUTPUT_DIR] [--no-plot] [--alpha ALPHA] [--outcome-type {continuous,binary}]
                                [--validation-proportion VALIDATION_PROPORTION] [--accelerator ACCELERATOR]
                                [--wandb-project WANDB_PROJECT] [--exposure-hidden [EXPOSURE_HIDDEN ...]]
                                [--exposure-max-epochs EXPOSURE_MAX_EPOCHS] [--exposure-batch-size EXPOSURE_BATCH_SIZE]
                                [--exposure-optimizer EXPOSURE_OPTIMIZER] [--exposure-learning-rate EXPOSURE_LEARNING_RATE]
                                [--exposure-weight-decay EXPOSURE_WEIGHT_DECAY] [--exposure-add-input-batchnorm]
                                [--outcome-hidden [OUTCOME_HIDDEN ...]] [--outcome-max-epochs OUTCOME_MAX_EPOCHS]
                                [--outcome-batch-size OUTCOME_BATCH_SIZE] [--outcome-optimizer OUTCOME_OPTIMIZER]
                                [--outcome-learning-rate OUTCOME_LEARNING_RATE] [--outcome-weight-decay OUTCOME_WEIGHT_DECAY]
                                [--outcome-add-input-batchnorm] --data DATA [--sep SEP] [--instruments [INSTRUMENTS ...]]
                                [--covariables [COVARIABLES ...]] --exposure EXPOSURE --outcome OUTCOME
                                [--genotypes-backend GENOTYPES_BACKEND] [--genotypes-backend-type GENOTYPES_BACKEND_TYPE]
                                [--sample-id-col SAMPLE_ID_COL]

optional arguments:
  -h, --help            show this help message and exit
  --n-gaussians N_GAUSSIANS
                        Number of gaussians used for the mixture density network.
  --exposure-network-type {mixture_density_net,gaussian_net,ridge}
                        Density model for the exposure network.
  --output-dir OUTPUT_DIR
  --no-plot             Disable plotting of diagnostics.
  --alpha ALPHA         Miscoverage level for the prediction interval.
  --outcome-type {continuous,binary}
                        Variable type for the outcome (binary vs continuous).
  --validation-proportion VALIDATION_PROPORTION
  --accelerator ACCELERATOR
                        Accelerator (e.g. gpu, cpu, mps) use to train the model. This will be passed to Pytorch Lightning.
  --wandb-project WANDB_PROJECT
                        Activates the Weights and Biases logger using the provided project name. Patterns such as project:run_name
                        are also allowed.
  --data DATA, -d DATA  Path to a data file.
  --sep SEP             Separator (column delimiter) for the data file.
  --instruments [INSTRUMENTS ...], -z [INSTRUMENTS ...]
                        The instrument (Z or G) in the case where we're not using genotypes provided through --genotypes. Multiple
                        values can be provided for multiple instruments. This should be column(s) in the data file.
  --covariables [COVARIABLES ...]
                        Variables which will be included in both stages.This should be column(s) in the data file.
  --exposure EXPOSURE, -x EXPOSURE
                        The exposure (X). This should be a column name in the data file.
  --outcome OUTCOME, -y OUTCOME
                        The outcome (Y). This should be a column name in the data file.
  --genotypes-backend GENOTYPES_BACKEND
                        Pickle containing a pytorch-genotypes backend. This can be created from various genetic data formats using
                        the 'pt-geno-create-backend' command line utility provided by pytorch genotypes.
  --genotypes-backend-type GENOTYPES_BACKEND_TYPE
                        Pickle containing a pytorch-genotypes backend. This can be created from various genetic data formats using
                        the 'pt-geno-create-backend' command line utility provided by pytorch genotypes.
  --sample-id-col SAMPLE_ID_COL
                        Column that contains the sample id. This is mandatory if genotypes are provided to enable joining.

Exposure Model Parameters:
  --exposure-hidden [EXPOSURE_HIDDEN ...]
  --exposure-max-epochs EXPOSURE_MAX_EPOCHS
  --exposure-batch-size EXPOSURE_BATCH_SIZE
  --exposure-optimizer EXPOSURE_OPTIMIZER
  --exposure-learning-rate EXPOSURE_LEARNING_RATE
  --exposure-weight-decay EXPOSURE_WEIGHT_DECAY
  --exposure-add-input-batchnorm

Outcome Model Parameters:
  --outcome-hidden [OUTCOME_HIDDEN ...]
  --outcome-max-epochs OUTCOME_MAX_EPOCHS
  --outcome-batch-size OUTCOME_BATCH_SIZE
  --outcome-optimizer OUTCOME_OPTIMIZER
  --outcome-learning-rate OUTCOME_LEARNING_RATE
  --outcome-weight-decay OUTCOME_WEIGHT_DECAY
  --outcome-add-input-batchnorm
```

</details>

## Bootstrapping and ensembling

To estimate confidence intervals, we often rely to bagging. Here we describe how to achieve this for the Quantile IV estimator. The other estimators may not support boostrapping yet. We rely on [GNU Parallel](https://www.gnu.org/software/parallel/) to fit the bootstrap resamples on an arbitrary number of devices.

First, we fit 50 Quantile IV models with bootstrap resampling (_i.e._ resampling the $n$ data points with replacement).

```
# Use -j to control the number of parallel jobs.
# Here, we use 20 jobs and consequentially 20 CPUs
#
# We recommend using --halt-on-error 2 so that parallel stops if one of the
# fit crashes with an error. This most often happens when there is a file not
# found error or other problems.
#
# For Quantile IV, the --fast flag avoids saving plots of the causal effect
# and other statistics that are irrelevant when we want to do bagging.
#
# We use the 'seq' command to generate an increasing list of integers 
# corresponding to the bootstrap iteration index.
parallel --halt-on-error 2 -j 20 ml-mr estimation quantile_iv \
  --data $filename \
  --sep ',' \
  --exposure exposure \
  --outcome outcome \
  --instruments v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 \
  --fast \
  --resample \
  --output "bs_"{} ::: $(seq 1 50)

# After fitting, the ensembling can be done using the API or, for example
# using the ml-mr evaluation module which will make plotting easy.
ml-mr evaluation \
  --true-function='lambda x: 0' \
  --plot --plot-filename my_ensemble_estimator.png \
  --input bs_* \
  --ensemble

```

To get a ensembling estimator instance using the API, the procedure looks like:
```python
from ml_mr.estimation.core import EnsembleMREstimator
from ml_mr.estimation.quantile_iv import QuantileIVEstimator

def get_ensemble_estimator(template, n, cls=QuantileIVEstimator):
    estimators = []
    for i in range(n):
        try:
            filename = template.format(i=i)
            estimator = cls.from_results(filename)
            estimators.append(estimator)

        except FileNotFoundError:
            pass

    return EnsembleMREstimator(*estimators)

# The returned object will support all of the methods (e.g. ate, cate,
# iv_reg_function) from MREstimator described above.
```

There is also a script in the ``ml-mr/scripts`` folder to facilitate plotting of ensemble estimators.

# Evaluation

When the true causal function is known, the evaluation module can be used to compare different instrumental variable estimates.

The format of the command is:

```
ml-mr  \
    evaluation \
    --input dfiv_estimate \
    --true-function 'lambda x: (0.3 * (x + 2) * (x - 1) + 1)' \
    --plot
```

Instead of a lambda, the ``--true-function`` argument can be a function in a python file:

```
ml-mr  \
    evaluation \
    --input dfiv_estimate \
    --true-function 'my_file.py:my_causal_function' \
    --plot
```

It is also possible to glob multiple estimates (which is especially useful in a sweep):

```
ml-mr  \
    evaluation \
    --input my_sweep/*/estimate* \
    --true-function 'my_file.py:my_causal_function' \
    --plot
```

A common practice is to focus on the central 95% of the exposure range. This can be done automatically using the ``--domain-95`` flag. If the user wishes to ensemble the provided estimators (_e.g._ if they were fit using bootstrap resamples and the user wants a simple bagging estimator) the ``--ensemble`` flag can be used.

## Metrics

Calling the evaluation module also computes metrics that are printed as a CSV file. The mean squared error is computed over a linear space over the observed data domain by default (_i.e._ min(exposure), max(exposure)). This means that we take evenly spaced exposure values between the minimum and observed maximum values, compute the true and estimated functions and take the mean of their squared difference.

The prediction interval width is computed similarly, but taking the mean of the width of the prediction interval over the domain.

If one is interested in the performance on a different space (_e.g._ 1st to 99th percentiles), it is currently necessary to use the ``MREstimator`` API directly.


## Meta keys

Estimators store many information in JSON format in a file called ``meta.json``. This file is useful to allow reloading the estimator from fitted models, but it also keeps track of validation metrics and other hyperparameters.

For example, here's a ``meta.json`` file from a fitted Quantile IV Model:

<details>
<summary>Display the contents of a meta.json file.</summary>

```json

{
  "q": 20,
  "output_dir": "estimate_run_46",
  "validation_proportion": 0.2,
  "fast": 1,
  "sqr": true,
  "exposure_hidden": [
    128,
    64
  ],
  "exposure_learning_rate": 0.0009258416040756616,
  "exposure_weight_decay": 0.0001,
  "exposure_batch_size": 10000,
  "exposure_max_epochs": 1000,
  "exposure_add_input_batchnorm": false,
  "outcome_hidden": [
    64,
    32
  ],
  "outcome_learning_rate": 0.000875839535511659,
  "outcome_weight_decay": 0.0001,
  "outcome_batch_size": 10000,
  "outcome_max_epochs": 1000,
  "outcome_add_input_batchnorm": false,
  "accelerator": "cpu",
  "wandb_project": null,
  "model": "quantile_iv",
  "domain": [
    -4.534079074859619,
    4.357717037200928
  ],
  "exposure_95_percentile": [
    -1.9370369940996168,
    1.9681992024183272
  ],
  "exposure_val_loss": 4.551474571228027,
  "outcome_val_loss": 0.7311554551124573,
  "q_hat": 7.812013626098633
}

```

</details>

To avoid having to parse this manually, we can specify fields to extract in the ``ml-mr evaluation`` command. For example, if you want the CSV file to include the ``q`` and ``outcome_val_loss``, we will use the ``--meta-keys q outcome_val_loss`` argument.

# Hyperparameter Sweep

Hyperparameter sweeps work by specifying a JSON configuration file  and calling

```
ml-mr sweep my_config.json
```

The configuration should define how to read a dataset, what algorithm to fit as well as configuration parameter samplers. 

To control the number of workers fitting models in parallel, use the ``--n-workers`` argument.

Simple examples can be found in this repository's "examples" directory.

To ensure that sampling is done as expected, it's possible to use the ``--create-db-only`` option. This will create a ``ml_mr_sweep_runs.db`` sqlite3 database without starting worker processes. You can then use sqlite3 to list all sampled parameter values and runs:

```sql
-- Connect using sqlite3 ml_mr_sweep_runs.db
-- You may also want to use .header on and .mode columns to make it easier
-- to read.
select * from run_parameters;
```

You can then start or resume the sweep using ``ml-mr sweep ml_mr_sweep_runs.db --n-workers N``.
