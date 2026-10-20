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
| Problem definition | `ProblemSpec`, `OutputSpec`, `ContinuousDomain`, `EnumeratedDomain`, `StructuredDomain`, `Box`, `InequalitySpec` |
| Session (ask/tell) | `Campaign` (`suggest`, `ingest`, `fail`, `cancel`, `outstanding`, `step`, `record`, `seed`, `load_history`, `recommend`, `result`, `state`, `current_model`, `scored_history`, `pending_count`, `add_hook`), `Candidate`, `CampaignResult`, `CampaignState` |
| Training targets | `Campaign(..., targets=)` / `Targets` (`(history, spec) -> Y` or `(Y, Yvar)`), `apply_targets` |
| Blocking driver | `run_campaign(spec, experiment, *, n_shots, batch_size, ...)`, `drive(campaign, experiment, evaluate, *, n_total, batch_size, on_persistent_fault, should_continue)` (the one blocking loop; `run_campaign` and the UI `RunController` run on it), `PersistentFaultError` |
| Spec-derived defaults | `default_policy`, `default_constructor`, `default_initial_designs`, `ProblemSpec.sample_designs` — what `Campaign(spec)` / `run_campaign` use when `policy` / `model_constructor` / `initial_designs` / `evaluate` are omitted |
| Policies | `Policy` (protocol: `suggest(history, spec, model, n, *, pending)` / `ingest`), `SimpleBO`, `ParetoBO`, `UncertaintySampling`, `BAXPolicy` (+ `candidates=`, `noise_at=`, `candidate_grid`, `collected_count`), `abbot.policy.condition_on_pending` |
| BAX goals | `Goal` (protocol), `PercentileSet`, `BandIntersection`, `Wishlist`, `FunctionGoal`, `LevelSet`, `TopK`, `goals_from_spec` |
| Surrogate seam | `ModelConstructor` (protocol: `build` + `provides_uq`), `SingleTaskGPConstructor`, `ModelListConstructor`, `EnsembleConstructor`, `default_constructor` |
| Experiment seams | `Experiment` (protocol: `run(design) -> outputs or observation`), `BatchExperiment` (protocol: `run_batch(designs) -> list`, for hardware that runs several at once), `Supervisor` (protocols), `Verdict`, `ManualExperiment`, `TensorExperiment`; mocks for offline runs: `MockAnalyticExperiment` (a seeded formula over any spec), `MockDatasetExperiment` (replays known results for a fixed set of designs; `.candidates`, `.from_history`) |
| Run data | `History`, `Record`, `DataStore`, `ResumeMismatchError` |
| Import/export | `abbot.io.spec_to_json` / `spec_from_json`, `history_to_csv` / `history_from_csv`, `spec_history_to_csv` / `spec_history_from_csv` |
| Live analysis | `abbot.analysis.convergence`, `hypervolume`, `recommended_set`, `parity`, `slice_1d`, `slice_2d`, `bax_target_region` |

### Provisional — may change
Useful and tested, but the surface is still settling; pin a version if you depend
on them.

- **`abbot.interop.*`** — the gest-api / Xopt field map (`spec_to_vocs`,
  `vocs_to_spec`, `CorePolicyGenerator`) and the SBI adapter (`PosteriorSampler`,
  `sbi_evaluate`, `TrainedPosterior`). These track external packages, which move.
- **`abbot.composite`** (`composite_evaluate`, `Rescore`, `ObjectiveReducer`) —
  composite objectives; the reducer-mount API may grow
  (e.g. reducing inside the acquisition).
- **`abbot.supervision`** — `per_shot`, `any_of`, `stall_monitor`, `drift_monitor`,
  `consecutive_fault_monitor`, `standard_supervisor`. The default thresholds were
  calibrated on sbido's scattering benchmark.
- **`abbot.ui.*`** — the NiceGUI app is an application, not a programmatic API. Drive
  the engine through `Campaign`, not by importing UI internals.
- **`abbot.acq_opt.*`** (`CandidateOptimizer`, `Continuous`, `DiscreteArgmax`,
  `default_optimizer`) — the
  seam is stable; the mixed-space story (`optimize_acqf_mixed`) is not yet wired, so
  signatures here may extend.

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
consumes, or a future constraint).

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
| `policy` | `default_policy(spec)`: `SimpleBO` for 1 objective, `ParetoBO` for 2+, with the optimizer the domain supports. No objectives (all `OBSERVE`) or a mixed space raises — pass a policy. |
| `model_constructor` | `default_constructor(spec)`: `SingleTaskGPConstructor` for 1 output, `ModelListConstructor` otherwise, normalized to the spec bounds. |
| `initial_designs` (`run_campaign`) | `default_initial_designs(spec)`: `max(2d + 1, 2m)` scrambled-Sobol points that respect enumerated values and inequality constraints. Ignored on resume. |
| `evaluate` (`run_campaign`) | identity — the experiment returns the outputs directly. |

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
  `ParetoBO` on a single-objective spec, a mixed continuous+enumerated space,
  a model whose uncertainty is too weak for the policy) raises immediately — it
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
- **Synchronous batches.** `Campaign.step(experiment, evaluate, n=k)` runs `k`
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
- Soft-deprecated aliases (`ObjectiveSpec`, `objectives=`, `_safe_fit`,
  `_ensure_grid`) are kept at least one minor version after a replacement ships.
- Breaking changes to Stable names happen only on a minor-version bump and are
  called out in the changelog.

## Planned, not yet built

Named here so you don't design around their absence as if permanent:
output-constraint enforcement for BO (feasibility-weighted acquisition over an
`OBSERVE` column), a run-level JSON serializer including the SBI `aux` payload, a
`TARGET` objective direction, mixed-space proposal (`optimize_acqf_mixed`),
continuous BAX (pathwise / seeded-ascent), the full `sbi`-backed benchmark, and
Bluesky interop.

- **Correlated multi-output `MultiTaskGP` — near-term, needs an ADR first.** The
  `ModelConstructor` seam is the right home, but it hinges on two decisions: where
  the `(n, m)`↔task-stacked bridge lives, and what the maximize-canonical sign flip
  means for a *learned cross-output covariance* (flipping a `MINIMIZE` column flips
  the sign of the correlation the model learns). Neither Xopt nor abbot ships this
  today, so it's a genuine capability, not parity work.
- **PFN / deep-kernel surrogates — F-tier research, same seam.** Both are a custom
  `ModelConstructor` returning a botorch `Model`; built when a problem needs them,
  not speculatively.
