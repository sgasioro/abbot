# abbot API — stability contract

This document is the **public API contract** for `abbot` 0.1.x. It states what you
can build on with confidence, what may still change, and what is internal. If a
name is importable from the top-level `abbot` package (see `abbot.__all__`), it is
part of the public surface described here.

Audience: people integrating abbot into a lab workflow, a notebook, or another
tool. For the *why* behind the architecture, see the top-level `README.md`.

## Stability tiers

### Stable — safe to build on
Breaking changes to these only on a minor-version bump before 1.0, and never
without a deprecation period.

| Area | Public surface |
| --- | --- |
| Problem definition | `ProblemSpec`, `OutputSpec`, `ContinuousDomain`, `IntegerDomain`, `EnumeratedDomain`, `StructuredDomain`, `Box`, `LinearConstraint`, `InequalitySpec` (soft-deprecated), `Source` (provisional, see below) |
| Session (ask/tell) | `Campaign` (`suggest`, `ingest`, `fail`, `cancel`, `outstanding`, `step`, `record`, `seed`, `load_history`, `recommend`, `result`, `state`, `current_model`, `scored_history`, `pending_count`, `add_hook`), `Candidate`, `CampaignResult`, `CampaignState` |
| Training targets | `Campaign(..., targets=)` / `Targets` (`(history, spec) -> Y` or `(Y, Yvar)`), `apply_targets` |
| Blocking driver | `run_campaign(spec, experiment, *, n_shots, batch_size, ...)`, `drive(campaign, experiment, evaluate, *, n_total, batch_size, on_persistent_fault, should_continue)` (the one blocking loop; `run_campaign` and the UI `RunController` run on it), `PersistentFaultError` |
| Spec-derived defaults | `default_policy`, `default_constructor`, `default_initial_designs`, `ProblemSpec.sample_designs` — what `Campaign(spec)` / `run_campaign` use when `policy` / `model_constructor` / `initial_designs` / `evaluate` are omitted |
| Policies | `Policy` (protocol: `suggest(history, spec, model, n, *, pending)` / `ingest`; optional `fixed=`, `last_acquisition_value`), `SimpleBO`, `ParetoBO`, `UncertaintySampling`, `BAXPolicy` (+ `candidates=`, `noise_at=`, `candidate_grid`, `collected_count`), `abbot.policy.condition_on_pending` |
| BAX goals | `Goal` (protocol; optional `bind(spec)` to resolve output names), `PercentileSet`, `BandIntersection`, `Wishlist`, `FunctionGoal`, `LevelSet`, `TopK`, `goals_from_spec` |
| Surrogate seam | `ModelConstructor` (protocol: `build` + `provides_uq`), `SingleTaskGPConstructor`, `ModelListConstructor`, `EnsembleConstructor`, `MultiFidelityGPConstructor` (provisional), `default_constructor` |
| Experiment seams | `Experiment` (protocol: `run(design) -> outputs or observation`, optional `cost(design)`), `BatchExperiment` (protocol: `run_batch(designs) -> list`, for hardware that runs several at once), `Supervisor` (protocols), `Verdict`, `ManualExperiment`, `TensorExperiment`; mocks for offline runs: `MockAnalyticExperiment` (a seeded formula over any spec; with sources, cheap runs are biased and `cost` simulates a cost ratio), `MockDatasetExperiment` (replays known results for a fixed set of designs; `.candidates`, `.from_history`) |
| Run data | `History`, `Record`, `DataStore`, `ResumeMismatchError` |
| Import/export | `abbot.io.spec_to_json` / `spec_from_json`, `history_to_csv` / `history_from_csv`, `spec_history_to_csv` / `spec_history_from_csv` |
| Live analysis | `abbot.analysis.convergence`, `hypervolume`, `recommended_set`, `parity`, `slice_1d`, `slice_2d`, `bax_target_region` |

### Provisional — may change
Useful and tested, but the surface is still settling; pin a version if you depend
on them.

- **`abbot.interop.*`** — the gest-api / Xopt field map (`spec_to_vocs`,
  `vocs_to_spec`), both directions of the generator bridge (`CorePolicyGenerator`:
  abbot in Xopt's loop; `GestGeneratorPolicy`: Xopt's generators in abbot's loop,
  see below) and the SBI adapter (`PosteriorSampler`, `sbi_evaluate`,
  `TrainedPosterior`). These track external packages, which move.
- **`abbot.composite`** (`composite_evaluate`, `Rescore`, `ObjectiveReducer`) —
  composite objectives; the reducer-mount API may grow
  (e.g. reducing inside the acquisition).
- **`abbot.supervision`** — `per_shot`, `any_of`, `stall_monitor`, `drift_monitor`,
  `consecutive_fault_monitor`, `standard_supervisor`, and the stopping supervisors
  `cost_cap`, `acquisition_monitor` (`Verdict.stop`). The default fault thresholds
  were calibrated on sbido's scattering benchmark.
- **`abbot.cost`** (`CostModel`, `FixedCost`, `CallableCost`, `FittedCost`,
  `PowerLawCost`, `SourceCost`), `Record.cost` / `Record.acquisition` and the timing
  fields, `SimpleBO` / `ParetoBO(cost_model=)` — new.
- **`abbot.ui.*`** — the NiceGUI app is an application, not a programmatic API. Drive
  the engine through `Campaign`, not by importing UI internals.
- **`abbot.acq_opt.*`** (`CandidateOptimizer`, `Continuous`, `DiscreteArgmax`,
  `default_optimizer`) — the seam is stable. `Continuous` now covers mixed spaces
  (integer inputs relaxed then rounded; enumerated inputs alongside continuous ones
  enumerated via `optimize_acqf_mixed`, up to 256 combinations), and `optimize` takes
  an optional `fixed_features={position: value}`.
- **Source variables** (`Source`, `ProblemSpec.sources`, `spec.design_names` /
  `source_names` / `target_features` / `at_target`, `CampaignResult.run_at_target` /
  `target_std`), `MultiFidelityGPConstructor`, `SimpleBO(choose_sources=,
  n_candidates=)`, `BAXPolicy(cost_model=, choose_sources=, source_levels=)` /
  `BAXPolicy.proposal_set`, and `.source_cost` on both — new.

### Private — do not touch
Anything with a leading underscore is internal and may change without notice,
including names that happen to be reachable (`Campaign._propose`,
`BAXPolicy._acquisition`, `BAXPolicy._collected`, `*._pending`, …). A few are
exercised by white-box tests; that is not a promise to external callers. Use the
public method with the same intent instead:

| If you reached for… | Use instead |
| --- | --- |
| `Campaign._safe_fit()` | `Campaign.current_model()` |
| `BAXPolicy._ensure_grid(spec)` | `BAXPolicy.candidate_grid(spec)` |
| `BAXPolicy._raw_mean_std(...)` / `._collected` / `._acquisition(...)` | `abbot.analysis.bax_target_region(policy, model, spec)` / `BAXPolicy.collected_count` |
| writing `Campaign._pending` / `._inflight` to inject a result | `Campaign.record(x, y)` |

(`_safe_fit` and `_ensure_grid` currently remain as thin deprecated aliases; they
will be removed in a future minor version.)

## Naming: `OutputSpec` / `outputs=` is canonical

A `ProblemSpec` is defined by its **outputs** — every measured quantity the
surrogate models (one GP column each, in `spec.outputs` order). Use:

```python
ProblemSpec(domain=[...], outputs=[OutputSpec(name="yield", direction="MAXIMIZE")])
```

`direction` is `"MAXIMIZE"` / `"MINIMIZE"` for an **optimized objective**, or
`"OBSERVE"` for a **modeled-but-not-optimized** output (a feature a BAX `Goal`
consumes, or an output that only has to stay within bounds).

- `ObjectiveSpec` and the `objectives=` constructor keyword are **soft-deprecated
  aliases**, kept for backward compatibility. They still work and will not be
  removed in 0.1.x, but new code and docs should use `OutputSpec` / `outputs=`.
- **Outputs vs. objectives are distinct, not aliases.** The *objectives* are the
  subset of outputs with a non-`OBSERVE` direction — the ones BO/MO optimize. So
  `spec.n_outputs` ≥ `spec.n_objectives`, and the derived members come in matching
  pairs: `output_names` / `objective_indices()`, `n_outputs` / `n_objectives`,
  `output_signs()` (all columns) / `objective_signs()` (the optimized subset). A
  pure-BAX spec is all-`OBSERVE`: `n_objectives == 0`, and `recommend()` returns
  `[]` (handover is the goal's job, not a single best point).

## Constraints

Constraints sit next to what they constrain, and are named by name:

```python
ProblemSpec(
    domain=[ContinuousDomain(name="temp", ...), ContinuousDomain(name="conc", ...)],
    outputs=[
        OutputSpec(name="yield", direction="MAXIMIZE"),
        OutputSpec(name="purity", direction="OBSERVE", min_val=0.9),
    ],
    domain_constraints=[LinearConstraint({"temp": 1, "conc": 2}, max_val=100)],
    output_constraints=[LinearConstraint({"yield": 1, "purity": -50}, min_val=0)],
)
```

- **A bound on one output** is `OutputSpec(min_val=..., max_val=...)`, in raw
  units. Any output can carry one; an `OBSERVE` output with bounds is a pure
  constraint.
- **`output_constraints`**: `LinearConstraint`s over output names, for bounds on a
  combination of outputs.
- **`domain_constraints`**: `LinearConstraint`s over design variable (or source)
  names. These are hard: no proposal or seed breaks them. `constraints=` is the old
  name and `InequalitySpec` (positions instead of names) the old form; both are
  still accepted.
- A name in the wrong list is an error that says which list it belongs in.

Output constraints are learned like the outputs: a design's chance of satisfying
them comes from the model (`abbot.feasibility`). What uses them:

| | With output constraints |
| --- | --- |
| `SimpleBO`, `ParetoBO` | Constrained log-NEI / log-NEHVI: improvement only counts where the constraints hold, weighted by the chance they do. Until a run at the target satisfies them (and without a `ref_point` for `ParetoBO`), proposals maximize the chance that a design will, favoring designs where that chance is uncertain. `ParetoBO` infers the reference point from the runs that satisfy them. |
| `SimpleBO` with sources | Entropy search weighted by the chance of satisfying the constraints at the target, with the best value sampled over designs likely to (an approximation). |
| `BAXPolicy` | The goal runs only on designs whose outputs satisfy the constraints: on the posterior mean (MeanBAX) and on each posterior sample (InfoBAX). So `TopK(k=5)` is the five best designs that satisfy them. `respect_constraints=False` lets the goal see every design, for bounds you want to map rather than filter by. When the constraints leave the goal nothing to select, `last_warning` says so (on the policy and the `Campaign`), and `bax_target_region` reports the goal's count before them as `before_constraints`. |
| `recommend` / `result()` | Only designs with at least a 50% chance (observed: that satisfied them) are ranked. If none, the design with the best chance (smallest observed violation) is returned, `CampaignResult.feasible` is `[False]`, and `Campaign.last_warning` says so. `CampaignResult.p_feasible` holds the chances. |
| `convergence`, `hypervolume` | Count only runs that satisfied the constraints. |
| VOCS, self-describing CSV | Output bounds round-trip. Linear constraints are refused rather than dropped (VOCS has none; use the JSON spec). |

`UncertaintySampling` ignores output constraints: it explores the model.

Independently of constraints: when the goal selects nothing on any posterior
sample, InfoBAX has no information to rank by, so it proposes by uncertainty and
says so in `last_warning` (before, it picked an arbitrary grid point).

## Names, not positions

The spec, its constraints and the goals refer to variables and outputs by name:

```python
ContinuousDomain(name="conc", lower=1e-4, upper=1.0, log=True)
Box({"yield": (0.4, 0.6)})                 # outputs left out are unbounded
TopK(k=5, output="yield"); LevelSet(0.9, output="purity"); PercentileSet(80, output="yield")
BandIntersection({"yield": (0.4, 0.6), "purity": (0.9, 1.0)})
Wishlist([{"yield": (0.4, 0.6)}, {"purity": (0.95, 1.0)}])
campaign.suggest(2, fixed={"temp": 30})
```

- **Positions still work** (`TopK(output=0)`, `Box(lower=[...], upper=[...])`,
  band lists with one entry per output). A goal with names is resolved by
  `goal.bind(spec)`, which `BAXPolicy` calls; using it unbound raises an error
  that says so. A custom goal can define `bind` the same way.
- **`log=True`** on a `ContinuousDomain` or `IntegerDomain` (positive bounds): the
  default models see `log10` of it, initial designs and BAX grids are log-uniform,
  and `Continuous` starts its restarts from log-uniform draws (when no enumerated
  input needs enumerating), so a narrow optimum in a low decade isn't missed. VOCS
  has no log scale: `spec_to_vocs` warns and exports it linearly. The CSV keeps
  it.
- **`fixed={name: value}`** on `Campaign.suggest` and `step` holds inputs at given
  values for those candidates (today's sample temperature, a setting you can't
  change); the policy chooses the rest. Values are checked against their
  domains. Queued seeds take the fixed values (redrawn if that breaks a domain
  constraint); a queued retry re-runs one design, so it waits for a `suggest` it
  matches. Fixing a source overrides the policy's choice of it. Every builtin
  policy accepts `fixed`; `BAXPolicy` proposes from its grid, so it can only fix
  integer or enumerated variables (the grid takes their allowed values) or values
  in a `candidates=` grid, and says so otherwise. A custom policy without `fixed=`
  is refused when it's used.

## Categories (labeled values)

```python
EnumeratedDomain(name="solvent", values=["water", "ethanol", "dmso"])
campaign.suggest(1)                               # [{"temp": 41.2, "solvent": "ethanol"}]
campaign.suggest(1, fixed={"solvent": "water"})
Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]), target="dft")
```

- **Numbers or labels.** An `EnumeratedDomain` with numbers is an ordered grid
  (nearby values are expected to behave alike). One with labels is a set of
  categories. The default models give categories a categorical kernel
  (BoTorch's `MixedSingleTaskGP`; inside the multi-fidelity models, a categorical
  kernel times the fidelity kernel), which learns how alike each pair of labels is
  instead of assuming an order.
- **Designs carry labels**: `suggest`, `outstanding`, `state().pending`,
  `experiment.run(design)`, `record({...}, y)`, `fixed=`, `seed([...])` /
  `initial_designs=[...]` (lists of dicts) and the CSVs. Tensors inside the loop
  (history records, `CampaignResult.designs`, policies, models) carry each label's
  **code**, its position in `values`. `spec.as_dict(x)` and `spec.as_tensor(design)`
  convert; `EnumeratedDomain.encode` / `decode` do one value.
- **`TensorExperiment(experiment, spec=spec)`** hands a vector experiment the codes;
  without `spec=` it refuses labels.
- **Domain constraints** can't name a labeled variable (a linear constraint can't
  weigh labels).
- **Sources** can be labeled; they are unordered, and `target` is a label.
- **Files:** a CSV enumerated row is labels if any value isn't a number. VOCS keeps
  values as a set, so `vocs_to_spec` sorts labels for a stable order. Changing the
  order of `values` changes the codes, so a store won't resume across it (the
  fingerprint differs).

## Xopt's generators in abbot's loop

```python
from abbot.interop.xopt import GestGeneratorPolicy
from xopt.generators.sequential.neldermead import NelderMeadGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

Campaign(spec, GestGeneratorPolicy(NelderMeadGenerator))
Campaign(spec, GestGeneratorPolicy(ExpectedImprovementGenerator, turbo_controller="optimize"))
```

`GestGeneratorPolicy` runs any gest-api generator (Nelder-Mead, RCDS, extremum
seeking, genetic algorithms, Xopt's BO with TuRBO or time dependence) under
abbot's loop, supervisors, store and UI. Pass a generator class (built on first
use from the spec, with your keyword arguments) or one built on
`GestGeneratorPolicy.vocs_for(spec)`.

- **`needs_model = False`.** A policy with its own model, or none, sets this, and
  the campaign skips its fit when proposing and passes `model=None`.
  Recommendations and the UI still fit abbot's model on the results.
- **Data.** Before each proposal the generator gets every clean record it hasn't
  seen. A resumed campaign hands a new generator the whole history.
- **Sequential generators** (Nelder-Mead, RCDS, extremum seeking) go one design at
  a time; several at once, or another while one is outstanding, is refused. They
  start from the best run they're given. A result they didn't propose (a
  `record` of an outside run) resets them, with a `last_warning`.
- **Labeled variables** reach the generator as codes, because Xopt's generators
  expect numbers.
- **Not supported:** `fixed=`, source variables, and what VOCS can't hold
  (domain and output linear constraints, integer domains).

## Outstanding candidates

`suggest(n)` returns `n` `Candidate`s: `{variable name: value}` dicts with an
`.id` (`"c0000"`, `"c0001"`, …). Each stays **outstanding** until it is closed, and
calling `suggest` again adds to the outstanding set rather than replacing it — so
a lab can have several vials, or a cluster several jobs, in flight at once.

| Close it with | Effect |
| --- | --- |
| `ingest(result, id=c.id)` | record the measured outputs (`ids=[...]` for several; with neither, one result per outstanding candidate, oldest first) |
| `fail(c.id, error)` | faulted record carrying `error`, retried up to `max_retries`, then halt |
| `cancel(c.id)` | withdraw it; nothing recorded, no retry |
| `record(x, y)` | if `x` exactly equals an outstanding design, closes it; otherwise records a new design |

`outstanding()` lists them, oldest first; `state().pending` / `.pending_ids` carry
the same in JSON. Every `Record` stores the `id` it closed. `step` runs the oldest
outstanding candidate before suggesting a new one.

**Proposals avoid work in flight.** When the policy proposes, it receives every
outstanding design (plus retries and seeds issued in the same call) as `pending`
`(k, d)`. `SimpleBO` / `ParetoBO` pass it to BoTorch as `X_pending`;
`UncertaintySampling` and `BAXPolicy` use `condition_on_pending(model, pending)`, a
kriging believer that conditions the model on the pending designs at its own mean
(mean unchanged, variance shrunk nearby). `BAXPolicy` also never re-proposes a
grid point that is measured or pending. A model that can't be conditioned (e.g. an
ensemble) is used as is, with a log line. A custom policy should accept
`pending=`; one that doesn't still works, with a warning that it may propose
duplicates.

**Resume is exact.** With a `DataStore`, outstanding candidates, queued retries,
unissued seeds and the id counter are written to `<root>/queue.json` after every
change. A `Campaign` built on the same store picks them up (ids included), and
`seed()` is ignored on resume.

**Resume refuses a different problem.** The store keeps `fingerprint.json`: the
spec, plus anything passed as `Campaign(..., fingerprint={...})` (config keys that
define "the same problem"). Resuming with a different one raises
`ResumeMismatchError` naming the changed keys; `allow_resume_mismatch=True` keeps
the old steps as valid history and records the new fingerprint.

## Column contract

**Designs are named.** `Campaign.suggest` returns, and `Experiment.run` receives, a
`{variable name: value}` dict in `spec.variable_names` order (`Campaign.record`
accepts one too). An experiment written against vectors — a vectorized simulator —
wraps in `TensorExperiment(exp)`, which hands `run` the `(d,)` tensor in the same
order (double, or `dtype=`).

`evaluate` / `Experiment.run` / `Campaign.ingest` / `Campaign.record` all speak
**raw physical units**: one value per output in `spec.outputs` order, or a
`{output name: value}` dict (any order; missing or unknown names are an error). The engine is *maximize-canonical*: it
applies `spec.output_signs()` internally so the surrogate always sees a
"bigger-is-better" space — you never pre-negate a `MINIMIZE` output. `History.outputs`
is `(n, n_outputs)` in the same order.

The two load-bearing conversions live on the spec so callers never re-assemble the
index/sign rule: `spec.objective_view(Y_raw)` selects the objective columns and
flips them to canonical; `spec.to_raw(canonical)` flips a canonical quantity back to
physical units (per output). Prefer these over hand-writing
`Y[:, objective_indices()] * objective_signs()`.

**Per-point noise (optional).** `ingest(..., output_vars=)` / `record(...,
output_var=)` accept a per-output noise *variance* `(m,)` per candidate (e.g. from
replicate measurements), stored on `Record.output_var`. When *every* clean record
carries it, the loop threads it to the surrogate's `train_Yvar` (a heteroskedastic
fit); otherwise the model infers the noise level. Variance is sign-invariant, so it
is never flipped by the canonical convention.

**Timing.** Every record logs `issued_at` (when `suggest` handed it out) and
`finished_at`; under `step` also `started_at` around the experiment run, so
`duration` is the run time (ask/tell only knows when the result came back). Proposed
records also carry abbot's own time: `fit_seconds` (the fit of the model the
proposal used) and `acquisition_seconds` (computing the proposal; shared by the
picks of one batch). Times survive a resume.

**Cost and stopping (optional).** Every record carries a `cost`: what the caller
passed (`ingest(..., costs=)`, `record(..., cost=)`, `fail(..., cost=)`:
core-hours, money); under `step`, what the experiment's optional `cost(design)`
reports after the run (core-hours from a scheduler, since wall-clock would include
queueing); else the run time `step` measured (a batch's time split evenly; abbot's
own fit and acquisition time is never counted).
`CampaignState` reports `costs` and `cost_spent`. `SimpleBO(cost_model=...)` and
`ParetoBO(cost_model=...)` maximize log-acquisition minus log cost (improvement per
unit cost), refitting the model to the recorded costs each step. `CallableCost`
takes a known formula; `FittedCost(fn, params)` learns a formula's parameters from
the recorded costs (least squares on log cost), and `PowerLawCost` is a ready-made
one for run time that scales steeply with a size-like variable. A supervisor's
`Verdict.stop` marks the campaign done without a fault: `cost_cap(budget)` stops
once `cost_spent` reaches the budget, and `acquisition_monitor(threshold, window)`
once the policy's recent acquisition values (stored per record from its
`last_acquisition_value`) all fall below `threshold`. `drive` ends normally on a
stop; ask/tell callers read `campaign.stopped` and may carry on.

**Source variables (fidelities, information sources).** `ProblemSpec(sources=[
Source(domain=IntegerDomain(name="atoms", lower=32, upper=512), target=256,
log=True)])` declares an input the experiment needs that isn't itself a design
choice: a fidelity knob (system size, mesh, convergence settings) or an information
source (two codes, simulation vs experiment; an unordered `EnumeratedDomain`).
Sources are inputs like any other: they follow the design variables in every design
vector, `suggest` hands them to the experiment in the design dict, and
`spec.variable_names` lists them (`spec.design_names` is the design-only view).
Declaring sources is enough to get multi-fidelity BO:
- `default_constructor` builds a `MultiFidelityGPConstructor`. Ordered sources get
  BoTorch's `SingleTaskMultiFidelityGP`, each source mapped to [0, 1] with its
  accurate end at 1 (log-scaled for `log=True`). Its default linear-truncated kernel
  models a run as the accurate result plus a bias that vanishes at the accurate end
  and varies with the design. Its `power=1` (a whole number) is fixed, not learned
  as in BoTorch: the kernel is only valid for whole-number powers, and a learned
  fractional one gives negative variances; `linear_truncated=False, iteration="name"` instead
  uses an exponential-decay kernel for a source whose results settle towards a
  limit. One unordered source gets a `MultiTaskGP` (noise shared across sources
  unless `Yvar` is given). Ordered and unordered sources together aren't supported
  yet.
- `SimpleBO` chooses sources: multi-fidelity max-value entropy search, the
  information a run gives about the best value at the target, per unit of cost.
  Batches are built one pick at a time. The cost is `cost_model=` if given, else a
  `SourceCost` learned from the recorded costs: a power law per positive ordered
  source, an exponential per other ordered source, a factor per level of an
  unordered source. It ignores the design variables, and it predicts a flat cost
  until there are enough costed records (with a warning if no run has a cost).
  `policy.source_cost` is the cost model in use. `SimpleBO(choose_sources=False)`
  keeps proposals at the target.
- `BAXPolicy`'s InfoBAX (also SwitchBAX's InfoBAX fallback) chooses sources. The goal
  still runs at the target; each grid design is also scored at a ladder of source
  levels (`source_levels=5` per ordered source, spaced on the model's scale, plus
  the target; all values of an unordered source; or `{name: [values]}`), by the
  information a run there gives about the goal's target set, per unit of cost (the
  same default `SourceCost`). `proposal_set(spec)` is the grid at every level. US,
  MeanBAX and SwitchBAX's MeanBAX phase propose at the target;
  `choose_sources=False` keeps InfoBAX there too.

Everything else works *at the target*:
- policies that don't choose sources (`ParetoBO`, `UncertaintySampling`, and the
  BAX strategies above) propose with the sources pinned at the target, and
  improvement is judged against the target;
- `recommend` scores each evaluated design at the target, counts a design run at
  several sources once (under its at-target record if any), and the observed
  fallback ranks only at-target runs; `result()` adds `run_at_target` and the
  model's `target_std`;
- default seeds are space-filling over every input, with two moved to the target.

`target` is a recommendation-time setting: it's left out of the resume fingerprint,
so it can change between runs. `ordered` defaults to yes for continuous / integer
sources and no for enumerated ones; an unordered source is enumerated, with
labels (`["dft", "xtb"]`, `target="dft"`) or whole numbers. `accurate="high"` (default) or `"low"` says which end of an
ordered source is more accurate (system size vs a tolerance), and `log=True` puts it
on a log scale for the model. The self-describing CSV, VOCS and the UI editor refuse
specs with sources; use the JSON spec.

**Swapping the surrogate.** The documented extension point is a custom
`ModelConstructor`: `build(X, Y, Yvar) -> botorch.Model`, with `provides_uq` set to
what the posterior actually supplies (`"calibrated_variance"` for a GP, `"samples"`
for an ensemble, `"point"` for a bare net — the loop enforces the match). A different
kernel, a fully-Bayesian GP, or a deep ensemble all slot in this way; see
`examples/custom_models.py`. `EnsembleConstructor` is the builtin non-GP example. The GP constructors take
`kernel="default"` (BoTorch's choice) or `kernel="matern52"` (scaled ARD Matern-5/2,
what multibax's sklearn GPs used).

## Defaults

Only a spec (and, for `run_campaign`, an experiment) is required. Omitted pieces
are derived from the spec:

| Omitted | Default |
| --- | --- |
| `policy` | `default_policy(spec)`: `SimpleBO` for 1 objective, `ParetoBO` for 2+, with the optimizer the inputs support (`DiscreteArgmax` if all enumerated, else `Continuous`). No objectives (all `OBSERVE`) raises — pass a policy. |
| `model_constructor` | `default_constructor(spec)`: `MultiFidelityGPConstructor` when the spec has sources, else `SingleTaskGPConstructor` for 1 output and `ModelListConstructor` otherwise, normalized to the spec bounds, with a log scale for `log=True` variables and a categorical kernel for labeled ones. |
| `initial_designs` (`Campaign`, `run_campaign`) | `default_initial_designs(spec)`: `max(2d + 1, 2m)` scrambled-Sobol points that respect enumerated values and domain constraints, two of them moved to the target when the spec has sources. `Campaign` queues them on its first `suggest`, unless the history isn't empty (a resume, or data you `record` first) or you `seed` your own; an empty tensor gives a cold start (random probes until a model can be fitted). Ignored on resume. |
| `evaluate` (`Campaign.step`, `drive`, `run_campaign`) | none — the experiment returns the outputs directly. |

The function names and signatures are Stable; the specific choices they make
(e.g. the seed count) may improve between minor versions. Pass the argument
explicitly if you need it pinned.

## BAX details

- **InfoBAX measures information about a noisy measurement.** Both entropy terms
  include the measurement noise at each candidate: the model's own (learned, or
  the mean per-point `Yvar`), or `BAXPolicy(noise_at=fn)` with `fn(X) -> (N, m)`
  variances in raw output units. The hallucinated target values are conditioned
  on with hyperparameters fixed and only a 1e-6 numerical jitter, since they are
  function values, not measurements.
- **US / MeanBAX compare outputs in relative units:** each output's posterior std
  is divided by its observed spread before averaging, so units don't decide.

## Failure behavior

- **Misconfiguration fails loud.** A policy that can't serve the spec (e.g.
  `ParetoBO` on a single-objective spec, too many enumerated combinations to
  enumerate, a model whose uncertainty is too weak for the policy) raises immediately — it
  will not silently do something else. The UQ guardrail reads the capability off
  the actual `ModelConstructor` (`provides_uq`), so it can't be bypassed by a
  mismatched argument — `Campaign` has no `model_uq` parameter.
- **Experiment errors retry, then halt.** If `experiment.run` raises inside
  `Campaign.step` / `run_campaign`, the shot is persisted as a faulted `Record`
  (`fault=True`, `outputs=None`, `error="<Type>: <message>"`) and the same candidate
  is retried up to `max_retries` times (default 1); after that the campaign halts
  and `run_campaign` raises `PersistentFaultError` ("fix the cause and rerun to
  resume"; `on_persistent_fault="continue"` spends the rest of the budget instead).
  The budget (`len(initial_designs) + n_shots`) is a total that every record
  counts against, failures and retries included, so the rerun finishes the
  remainder. Errors from `evaluate` propagate unchanged — they are analysis bugs,
  and re-running the experiment would not fix them.
- **Pre-run check.** `Campaign(..., pre_run=check)` / `run_campaign(..., pre_run=)`
  calls `check(candidate_dict)` before each `step` runs the experiment. Return
  `None` to proceed or a reason string to block: the campaign halts with that
  reason and the candidate is left outstanding (in `state().pending`), not run. Use it
  for safety interlocks; soft feasibility rules belong in the spec's constraints.
- **Synchronous batches.** `Campaign.step(experiment, evaluate=None, n=k)` runs `k`
  designs per step; `run_campaign(..., batch_size=k)` / `drive(..., batch_size=k)`
  step that way, shrinking the last batch so the budget (still counted in designs)
  isn't overshot. A `BatchExperiment` gets every step in one `run_batch(designs)`
  call, batches of one included (`TensorExperiment` forwards it); an `Experiment`
  gets `run` once per design. A slot of `run_batch`'s result that is an
  exception fails just that design; `run_batch` raising fails the whole batch.
  Failed designs retry as above, the batch's other results are recorded, and
  `pre_run` must pass every design before any runs. `on_step` hooks fire once per
  step, after the whole batch is recorded. Picks within a batch are
  pending-aware (BO: joint q-batch; BAX / US: greedy kriging believer).
- **Transient numerical hiccups self-heal.** A one-off GP/acquisition failure mid-run
  falls back to a single random probe and records a message on
  `Campaign.last_warning` (surfaced as a toast in the UI); the next step retries
  normally. Unattended runs survive blips without going silently wrong.

## Deprecation policy (pre-1.0)

- `abbot` is `0.1.0` — pre-release. The **Stable** tier above is the compatibility
  promise; **Provisional** may change on any release; **Private** has no promise.
- Soft-deprecated aliases (`ObjectiveSpec`, `objectives=`, `constraints=`,
  `InequalitySpec`, `_safe_fit`, `_ensure_grid`) are kept at least one minor version after a replacement ships.
- Breaking changes to Stable names happen only on a minor-version bump and are
  called out in the changelog.

## Planned, not yet built

Named here so you don't design around their absence as if permanent:
output constraints in the UI, a feasibility stopping rule, a run-level JSON serializer including the SBI `aux` payload, a
`TARGET` objective direction, continuous BAX (pathwise / seeded-ascent), the full
`sbi`-backed benchmark, and a blop backend for Bluesky.

- **Correlated multi-output `MultiTaskGP` — near-term, needs an ADR first.** The
  `ModelConstructor` seam is the right home, but it hinges on two decisions: where
  the `(n, m)`↔task-stacked bridge lives, and what the maximize-canonical sign flip
  means for a *learned cross-output covariance* (flipping a `MINIMIZE` column flips
  the sign of the correlation the model learns). Neither Xopt nor abbot ships this
  today, so it's a genuine capability, not parity work.
- **PFN / deep-kernel surrogates — F-tier research, same seam.** Both are a custom
  `ModelConstructor` returning a botorch `Model`; built when a problem needs them,
  not speculatively.
