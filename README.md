# PyCIB Cross-Impact Balance (CIB) Analysis Package

PyCIB is a Python implementation of Cross-Impact Balance analysis with uncertainty quantification and robustness testing capabilities.

## Overview

A Python implementation of Cross-Impact Balance (CIB) analysis is provided for scenario construction and evaluation from expert-elicited cross-impacts. Deterministic workflows are supported for consistency checking, succession to attractors, and scenario enumeration where feasible. Uncertainty and stochasticity are supported so that probabilistic outputs can be produced from Monte Carlo ensembles and branching pathway graphs (simulation-first probabilistic CIB). Workshop judgements may be treated as uncertain through confidence-coded sampling of the cross-impact matrix (CIM), and uncertainty may be scaled with horizon where such a modelling assumption is required. Robustness may be assessed under structural shocks, including correlated and heavy-tailed variants where configured. Within-period succession may be perturbed by dynamic shocks with persistence, including rare-event components where configured. Multi-period pathways may be simulated under threshold rules and cyclic descriptors, and equilibrium analogues may be reported where configured. Probabilistic summaries may be reported as per-period state probabilities, quantiles for mapped numeric outcomes, and explicit pathway structures in branching graphs. A separate probabilistic cross-impact analysis (CIA) extension, `cib.prob`, is also provided. In this extension, an explicit joint distribution is fitted from user-specified marginal probabilities and cross-impact multipliers interpreted as conditional probability ratios. This fitted joint distribution is not derived from CIB impact balances unless it has been parameterised explicitly as such. Expert judgements may be aggregated with weights (including partial coverage), and network analysis is provided for structural interpretation and communication. The package is intended for scenario work in settings such as energy transition planning, urban development, technology assessment, and policy analysis.

The following is provided by this package:

- **Phase 1**: Deterministic CIB analysis with consistency checking,
  succession operators, and scenario enumeration
- **Phase 2**: Uncertainty quantification via confidence-coded impacts and
  Monte Carlo estimation, plus robustness testing under structural shocks
- **Optional extensions**: expert aggregation (weighted, partial coverage) and
  simulation-first dynamic (multi-period) CIB with thresholds/cycles.

## Installation

```bash
cd pycib
pip install -e .
```

## Quick Start

### Basic Deterministic CIB

```python
from cib import CIBMatrix, ScenarioAnalyzer

# Define descriptors and states
descriptors = {
    'Tourism': ['Decrease', 'Increase'],
    'Urban_Structure': ['Densification', 'Sprawl'],
    'GDP_Growth': ['Weak', 'Strong']
}

# Create matrix
matrix = CIBMatrix(descriptors)

# Set impact values
matrix.set_impact('Tourism', 'Increase', 'GDP_Growth', 'Strong', 2)
# ... set all impacts

# Find consistent scenarios
analyser = ScenarioAnalyzer(matrix)
consistent_scenarios = analyser.find_all_consistent()

print(f"Found {len(consistent_scenarios)} consistent scenarios")
```

### Feasibility constraints (domain rules)

When domain rules are required to be respected (in addition to CIB consistency), feasibility constraints may be specified.

```python
from cib import CIBMatrix, ScenarioAnalyzer
from cib.constraints import AllowedStates, ForbiddenPair, Implies
from cib.solvers.config import ExactSolverConfig

matrix = CIBMatrix(descriptors)
# Impacts are set as usual.

analyser = ScenarioAnalyzer(matrix)
res = analyser.find_all_consistent_exact(
    config=ExactSolverConfig(
        constraints=[
            Implies("Electrification_Demand", "High", "Renewables_Deployment", "Moderate"),
            ForbiddenPair("Policy_Stringency", "High", "Public_Acceptance", "Low"),
            AllowedStates("Technology_Costs", allowed={"Moderate", "Low"}),
        ]
    )
)
print(len(res.scenarios))
```

### Sparse scoring backend (Monte Carlo succession)

When a large, sparse impact structure is provided, a sparse scoring backend may be selected for Monte Carlo attractor discovery:

```python
from cib.solvers.config import MonteCarloAttractorConfig

cfg = MonteCarloAttractorConfig(runs=5000, fast_backend="sparse")
res = analyser.find_attractors_monte_carlo(config=cfg)
print(res.status, len(res.counts))
```

### Dynamic CIB (5-state demonstration: probability bands + fan + spaghetti, fat tails + jumps)

```python
import matplotlib.pyplot as plt

from cib import (
    DynamicCIB,
    DynamicVisualizer,
    UncertainCIBMatrix,
    numeric_quantile_timelines,
    state_probability_timelines,
)
from cib.example_data import (
    DATASET_B5_CONFIDENCE,
    DATASET_B5_DESCRIPTORS,
    DATASET_B5_IMPACTS,
    DATASET_B5_INITIAL_SCENARIO,
    DATASET_B5_NUMERIC_MAPPING,
    dataset_b5_cyclic_descriptors,
    dataset_b5_threshold_rule_fast_permitting,
)
from cib.shocks import ShockModel

periods = [2025, 2030, 2035, 2040, 2045]
descriptor = "Electrification_Demand"

matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)

dyn = DynamicCIB(matrix, periods=periods)
for cd in dataset_b5_cyclic_descriptors():  # exogenous drift/inertia for selected drivers
    dyn.add_cyclic_descriptor(cd)
dyn.add_threshold_rule(dataset_b5_threshold_rule_fast_permitting())

# Non-Gaussian dynamic shocks: Student-t innovations + rare jumps.
n_runs = 500
base_seed = 123
paths = []
for m in range(n_runs):
    from cib.example_data import seeds_for_run

    seeds = seeds_for_run(base_seed, m)
    sm = ShockModel(matrix)
    sm.add_dynamic_shocks(
        periods=periods,
        tau=0.26,
        rho=0.6,
        innovation_dist="student_t",
        innovation_df=5.0,
        jump_prob=0.02,
        jump_scale=0.70,
    )
    dynamic_shocks = sm.sample_dynamic_shocks(seeds["dynamic_shock_seed"])
    sigma_by_period = {int(t): 1.0 + 0.85 * i for i, t in enumerate(periods)}
    paths.append(
        dyn.simulate_path(
            initial=DATASET_B5_INITIAL_SCENARIO,
            seed=seeds["dynamic_shock_seed"],
            dynamic_shocks_by_period=dynamic_shocks,
            judgment_sigma_scale_by_period=sigma_by_period,
            structural_sigma=0.15,
            structural_seed_base=seeds["structural_shock_seed"],
            equilibrium_mode="relax_unshocked",
        )
    )

timelines = state_probability_timelines(paths, scenario_mode="realized")
timelines_equilibrium = state_probability_timelines(paths, scenario_mode="equilibrium")
quantiles = numeric_quantile_timelines(
    paths,
    descriptor=descriptor,
    numeric_mapping=DATASET_B5_NUMERIC_MAPPING[descriptor],
    quantiles=(0.05, 0.5, 0.95),
    scenario_mode="realized",
)

mapping = DATASET_B5_NUMERIC_MAPPING[descriptor]
expectation = {
    int(t): sum(float(p) * float(mapping[s]) for s, p in timelines[t][descriptor].items())
    for t in timelines
}

DynamicVisualizer.plot_descriptor_stochastic_summary(
    timelines=timelines,
    quantiles_by_period=quantiles,
    numeric_expectation_by_period=expectation,
    descriptor=descriptor,
    title="Electrification_Demand (5-state): probability bands + fan + spaghetti",
    spaghetti_paths=paths,
    spaghetti_numeric_mapping=mapping,
    spaghetti_max_runs=200,
)
plt.tight_layout()
plt.show()
```

### Monte Carlo vs branching (summary)

Detailed guidance on when to use Monte Carlo ensembles versus branching pathway graphs is documented in `docs/Documentation.md`.

### Diagnostics, attribution, and sensitivity (summary)

For research workflows where interpretability and robustness are required, additional utilities are provided:

- **Scenario diagnostics**: consistency, margins, and “brink” descriptors are provided by `scenario_diagnostics`.
- **Local attribution**: margin-to-switching is decomposed by source contributions by `attribute_scenario`, and simple bounded flip candidates are suggested by `flip_candidates_for_descriptor` (heuristic).
- **Rare-event diagnostics**: basic reliability summaries are provided by `event_rate_diagnostics` and `near_miss_rate`.
- **Global sensitivity**: an ensemble-level driver–outcome sensitivity report is provided by `compute_global_sensitivity_dynamic`.

## Documentation

Detailed usage examples are provided in the `examples/` directory:

- `dynamic_cib.ipynb`: Canonical 5-state dynamic example (probability bands + fan + spaghetti)
- `example_dynamic_cib_c10.py`: Dynamic CIB on `DATASET_C10` (workshop-scale dataset)
- `example_dynamic_cib_c15_rare_events.py`: Dynamic CIB on `DATASET_C15` (rare events + regime switching)
- `example_state_binning.py`: State binning (model reduction for large state spaces)
- `example_solver_modes.py`: Scaling solver modes (exact pruned enumeration and Monte Carlo attractors)
- `example_solver_modes_c10.py`: Scaling solver modes on `DATASET_C10` (workshop-scale dataset)
- `example_enumeration_c10.py`: Full enumeration on `DATASET_C10` (brute-force consistency filtering + exact solver parity; a diagnostic plot is written to `results/example_enumeration_c10_plot_1.png`)
- `example_attractor_basin_validation_c10.py`: Exact basin weights on `DATASET_C10` (full-space succession) with a Monte Carlo comparison plot written to `results/example_attractor_basin_validation_c10_plot_1.png`

Additional docs:

- `docs/api_reference.md`: high-level API surface
- `docs/Documentation.md`: concise equations, modelling choices, and result interpretation (simulation-first CIB and joint-distribution probabilistic CIA, `cib.prob`)

## Key Features

### Phase 1: Core Deterministic CIB

- CIBMatrix: Store and manipulate cross-impact relationships
- Scenario: Represent state assignments
- ConsistencyChecker: Validate scenario consistency
- Succession operators: Find consistent scenarios iteratively
- ScenarioAnalyzer: Enumerate and filter scenarios
- Import/Export: CSV and JSON file support

### Phase 2: Uncertainty and Robustness

- UncertainCIBMatrix: Confidence-coded impacts with uncertainty modelling
- MonteCarloAnalyzer: Estimate P(consistent | z) via Monte Carlo
- ShockModel: Apply structural perturbations to impact matrices
- RobustnessTester: Evaluate scenario stability under shocks

## Example Datasets

An example dataset is provided in `cib.example_data`:

- **Energy transition demonstration**: 5 descriptors × 5 states (canonical example)

Complete impact matrices and confidence codes are included in the dataset.

Preferred import path for datasets is `cib.example_data`.

## Results and interpretation

Generated outputs and guidance on interpretation are documented in `docs/Documentation.md` (including `results/` file descriptions, scenario diagnostics, branching trees, and scenario network plots).

## Coding

OpenAI was utilised as a language and code assistant during the preparation of this work.

## Testing

The test suite may be run with:

```bash
cd pycib
python3 -m pytest tests/
```

## Licence

Details are provided in the LICENSE file.

### Cite as:

Ross, A. G. (2025). PyCIB Cross-Impact Balance (CIB) Analysis Package. [10.5281/zenodo.18367511](https://doi.org/10.5281/zenodo.18367511)  

## References

Aoki, M., & Yoshikawa, H. (2011). Reconstructing macroeconomics: a perspective from statistical physics and combinatorial stochastic processes. Cambridge University Press.

Baqaee, D. R., & Farhi, E. (2019). The macroeconomic impact of microeconomic shocks: Beyond Hulten's theorem. Econometrica, 87(4), 1155-1203.

Jermann, U., & Quadrini, V. (2012). Macroeconomic effects of financial shocks. American Economic Review, 102(1), 238-271.

Kearney, N. M. (2022). Prophesy: A new tool for dynamic CIB.

Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series methods for empirical macroeconomics. Foundations and Trends in Econometrics, 3(4), 267-358.

Weimer-Jehle, W. (2006). Cross-impact balances: A system-theoretical approach to cross-impact analysis. Technological Forecasting and Social Change, 73(4), 334-361.

Weimer-Jehle, W. (2023). Cross-Impact Balances (CIB) for Scenario Analysis. Cham, Switzerland: Springer.

Roponen, J., & Salo, A. (2024). A probabilistic cross‐impact methodology for explorative scenario analysis. Futures & Foresight Science, 6(1), e165.

Salo, A., Tosoni, E., Roponen, J., & Bunn, D. W. (2022). Using cross‐impact analysis for probabilistic risk assessment. Futures & foresight science, 4(2), e2103.

Vögele, S., Poganietz, W. R., & Mayer, P. (2019). How to deal with non-linear pathways towards energy futures: Concept and application of the cross-impact balance analysis. TATuP–Journal for Technology Assessment in Theory and Practice, 28(3), 20-26.
