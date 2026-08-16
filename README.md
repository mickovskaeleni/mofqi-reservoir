# mofqi-reservoir

A Python implementation of batch-mode Fitted Q-Iteration (FQI) and Multi-Objective Fitted Q-Iteration (MOFQI) for reservoir-operation problems.

The package implements the methodological core of:

> Pianosi, F., Castelletti, A. and Restelli, M. (2013). Tree-based fitted Q-iteration for multi-objective Markov decision processes in water resource management. *Journal of Hydroinformatics*, 15(2), 258–270.
> https://doi.org/10.2166/hydro.2013.169

## Scope

The package provides:

- validated offline state-action transition batches;
- vector-valued rewards for multi-objective problems;
- Bellman-target construction for batch FQI;
- iterative FQI using scikit-learn Extremely Randomized Trees;
- objective-preference validation and sampling;
- linear scalarisation of vector rewards;
- weight augmentation of the MOFQI training dataset;
- a single weight-conditioned MOFQI training process;
- greedy policy extraction for FQI and MOFQI;
- simulation-based policy evaluation with separate objective returns;
- comparison of repeated FQI training with single-process MOFQI;
- a synthetic two-objective reservoir environment;
- an executable Jupyter notebook demonstrating flood-irrigation trade-offs.

## Installation

Python 3.11 or newer is required.

Install the package and its core dependencies:

```bash
python -m pip install -e .
```

Install the development, testing and notebook dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from mofqi_reservoir import (
    SyntheticReservoir,
    evaluate_mofqi_policy,
    sample_reservoir_transitions,
    train_fqi_mofqi_comparison,
)

batch = sample_reservoir_transitions(
    n_samples=1_000,
    random_state=42,
)

flood_weights = np.linspace(0.0, 1.0, 11)
weights = np.column_stack(
    (flood_weights, 1.0 - flood_weights)
)

candidate_actions = np.linspace(0.0, 160.0, 33)

comparison = train_fqi_mofqi_comparison(
    batch=batch,
    weights=weights,
    candidate_actions=candidate_actions,
    gamma=1.0,
    n_iterations=10,
    n_estimators=20,
    random_state=42,
    n_jobs=-1,
)

environment = SyntheticReservoir(
    initial_storage=50.0,
    episode_length=10,
    random_state=42,
)

evaluation = evaluate_mofqi_policy(
    environment=environment,
    learner=comparison.mofqi_learner,
    weights=[0.5, 0.5],
    max_steps=10,
)

print(evaluation.objective_returns)
```

For the synthetic reservoir, preference weights are ordered as:

```text
[flood weight, irrigation weight]
```

Higher objective returns are better. Both rewards are negative penalties, so values closer to zero are better.

## Main API

### Transition data

- `TransitionBatch`: validated transitions with scalar rewards.
- `MultiObjectiveTransitionBatch`: validated transitions with one reward per objective.

### Fitted Q-Iteration

- `FittedQIteration`: iterative batch FQI using Extra Trees.
- `build_bellman_targets`: construct regression targets for one FQI iteration.
- `select_actions`: extract greedy actions from a fitted FQI learner.

### Multi-Objective Fitted Q-Iteration

- `MultiObjectiveFittedQIteration`: fit one weight-conditioned MOFQI process.
- `validate_preference_weights`: validate weights on the unit simplex.
- `sample_preference_weights`: sample reproducible preference weights.
- `scalarize_rewards`: calculate weighted sums of vector rewards.
- `augment_transitions`: append independent preference coordinates to states and scalarise rewards.

### Policy evaluation

- `PolicyEvaluation`: stores states, actions, rewards and objective returns from a rollout.
- `evaluate_policy`: evaluate a generic policy in an environment.
- `evaluate_fqi_policy`: evaluate a fitted FQI policy.
- `evaluate_mofqi_policy`: evaluate a fitted MOFQI policy for a requested preference.

### Synthetic reservoir

- `SyntheticReservoir`: stationary stochastic reservoir with flood and irrigation objectives.
- `sample_reservoir_transitions`: generate a reproducible offline transition dataset.

### Method comparison

- `TrainingComparison`: stores repeated-FQI learners, the MOFQI learner and training times.
- `train_fqi_mofqi_comparison`: train repeated scalar FQI models and one MOFQI process over the same preference weights.

## Synthetic reservoir case study

The executable notebook is located at:

```text
notebooks/synthetic_reservoir_case_study.ipynb
```

Start JupyterLab from the repository root:

```bash
jupyter lab notebooks/synthetic_reservoir_case_study.ipynb
```

The notebook:

1. generates reproducible offline reservoir transitions;
2. visualises the flood and irrigation reward functions;
3. trains repeated FQI models and one MOFQI model;
4. evaluates policies for several objective-preference combinations;
5. plots the evaluated objective trade-offs;
6. compares the measured training times; and
7. visualises preference-conditioned reservoir operating policies.

## Validation

Run the automated test suite with:

```bash
pytest
```

The tests cover:

- transition validation and dimensional consistency;
- Bellman-target construction;
- iterative FQI fitting and prediction;
- objective-weight validation and reward scalarisation;
- MOFQI transition augmentation and fitting;
- greedy policy extraction;
- policy evaluation;
- reservoir dynamics and reproducibility;
- repeated-FQI versus MOFQI comparison; and
- package-level imports.

GitHub Actions runs the test suite on Python 3.11 and Python 3.13.

## Methodological limitations

- The implementation uses scikit-learn's standard `ExtraTreesRegressor`; it does not reproduce the paper's customised tree-pruning procedure.
- The demonstration notebook uses reduced sample sizes and a coarser action grid so that it can run on a personal computer.
- The notebook is a methodological reproduction and does not claim to reproduce the paper's exact numerical results.
- The evaluated trade-off points are not guaranteed to be non-dominated or to form an accurate Pareto frontier under the compact experimental configuration.
- Training-time measurements are illustrative rather than a formal benchmark and depend on hardware, parallel execution, dataset augmentation and model configuration.
- Linear scalarisation may not recover all policies on a non-convex Pareto frontier.
- The synthetic environment is intended for method demonstration, not operational reservoir management.

## Project structure

```text
src/mofqi_reservoir/    Package source code
tests/                  Automated tests
notebooks/              Executable case-study notebook
.github/workflows/      Continuous-integration configuration
```

## Author

Eleni Mickovska

## License

This project is licensed under the MIT License.
