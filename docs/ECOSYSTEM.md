# abbot in the ecosystem

Where abbot sits next to BoTorch, Ax, Xopt and Bluesky/blop, what it owns, and
what it deliberately leaves to them. Neighbor details were checked against
ax-platform 1.3.1, xopt 3.2.2 and blop 1.1.0 (October 2026); they move, so
re-check before relying on a specific claim.

## Position

abbot is a standalone decision engine for experiments whose **goal isn't an
optimum** (BAX goals: a level set, the best point per band, every condition in
spec) or whose **measurement isn't a number** (a spectrum, an SBI posterior,
replicates reduced by an `evaluate`). It runs a supervised ask/tell loop that
survives flaky experiments, and ships a lab UI light enough to start a campaign
in minutes.

It plugs into the orchestrators people already use through thin adapters in
`abbot.interop`, as a peer rather than a component. Algorithms that fit another
project naturally may be offered upstream; nothing is written on the assumption
that it will be.

## Layers

| Layer | Owner | abbot's relationship |
|---|---|---|
| Models, acquisition functions, acquisition optimizers | BoTorch | Built on it; adds little to the numerics |
| Campaign: problem spec, goals, ask/tell with ids, outstanding and retry queues, supervision, persistence, lab UI | **abbot** | This is abbot |
| Running evaluations: parallel evaluators, devices, plans, queues | Xopt (evaluator, Badger GUI), Bluesky (RunEngine, queue server), or abbot's own blocking loop and UI | abbot proposes; they execute |

## Neighbors

**BoTorch.** A library, not a loop: no problem spec, candidate ids, queue,
resume or fault handling. abbot adds the campaign layer, the BAX family and
composite measurements on top.

**Ax.** The closest overlap. Its `Client` covers experiment setup, trials,
failure marking, early stopping, JSON/database snapshots and Pareto fronts, with
a large team behind it. For single- or multi-objective optimization of scalar
measurements, Ax is a good answer. abbot still covers standard BO (`SimpleBO`,
`ParetoBO`, multi-fidelity) so a lab never has to leave it: the
overhead of switching tools pushes people to a hand-rolled BoTorch script, not
to Ax. Ax's model is
objectives and trials; BAX goals, posterior-sample algorithm execution and raw
observations are outside that model.

**Xopt.** Accelerator- and beamline-driven. Problems are a VOCS, data is a
DataFrame of scalars, generators `generate(n)` from it. Its `BaxGenerator` is
InfoBAX only (expected information gain over `GridOptimize` /
`CurvatureGridOptimize`), without discrete variables. abbot's BAX is broader
(InfoBAX, MeanBAX, SwitchBAX; arbitrary goals; finite candidate sets; greedy
kriging-believer batches). Interop runs both ways over the gest-api
`suggest`/`ingest` protocol, with `spec_to_vocs` / `vocs_to_spec` moving the
problem itself:
- abbot's proposals in Xopt's loop: `CorePolicyGenerator` wraps a `Campaign` as
  a gest `Generator`, so an abbot problem runs under `Xopt(generator=...)` on
  Xopt's parallel evaluators, Badger and control systems;
- Xopt's proposals in abbot's loop: `GestGeneratorPolicy` wraps any
  gest generator (TuRBO, Nelder-Mead, RCDS, genetic algorithms, time-dependent
  BO) as an abbot `Policy`, under abbot's supervisors, resume, cost tracking and
  UI.

**Bluesky / blop.** Bluesky runs the hardware; blop is its optimization agent,
used at several light sources. blop defines a small `Optimizer` protocol
(`suggest(n)` with `_id`s, `ingest`, `get_best_points`, plus optional
`register_failures`, `should_stop`, `checkpoint`) with Ax (default), Xopt and
scipy backends. It maps nearly one-to-one onto `Campaign`, so abbot can be a
blop backend; that is the route onto Bluesky beamlines, not a separate
bluesky-adaptive agent competing with blop. blop requires Bluesky even in its
core install, so it is not a home for labs that don't run a RunEngine.

## Why not extend one of them

- **Xopt**: composite measurements and per-shot supervision need changes to its
  data model, not a new generator, and its focus is accelerators and beamlines.
  Interop yes; housing every lab use case there is not its aim.
- **Ax**: built around objectives and trials, which BAX goals and raw composite
  observations don't map onto.
- **blop**: Bluesky-only by design.

The BAX algorithms on their own are portable and could live elsewhere. What
needs abbot is owning the loop: raw observation → `evaluate` → outputs with the
observation kept; candidate ids, the outstanding queue, retry/halt, supervisors
and resume; and a UI that needs no EPICS, Bluesky or Badger.

## Build or borrow

From comparing abbot with these neighbors (October 2026). The rule:
build natively what has to work with abbot's own pieces (the spec, sources and
cost, BAX goals, recommendation at the target, guardrails, the UI); borrow
algorithms that are self-contained and run on their own data.

| Need | Route |
|---|---|
| Output constraints, named spec references, string categories, log-scaled design variables, pinning a variable at proposal time | Native: they are the spec, and constraints must work with BAX, multi-fidelity and recommendation |
| Model diagnostics, stopping on a target or on no improvement | Native: small, and part of the loop |
| Per-variable travel limits (hardware safety) | Native: they wrap every policy, BAX and multi-fidelity included |
| Physics prior mean | Native: `default_constructor(spec, prior_mean=...)` |
| Correlated outputs | Native: `default_constructor(spec, correlated=True)`, on BoTorch's `MultiTaskGP` with abbot's wide-format bridge, per-output scaling and an unconstrained output covariance |
| Retries that end with a person, and learned bad regions (runs that give nothing usable) | Native: `Campaign.resolve`, `Verdict(bad_region=)`, a usability classifier every policy and the recommendation use |
| Many inputs, few that matter | TuRBO through `GestGeneratorPolicy`, or a neural-network surrogate (`EnsembleConstructor`); `campaign.check()` points there. SAAS (BoTorch, needs JAX) only on demand, through the `ModelConstructor` seam |
| TuRBO, non-BO optimizers (Nelder-Mead, RCDS, extremum seeking, genetic algorithms), time-dependent BO, MGGPO | Xopt, through `GestGeneratorPolicy` |
| Parallel and asynchronous evaluation at scale, control systems, Badger | Xopt, through `CorePolicyGenerator` |
| Bluesky devices, plans, queue server, moving to the best setting, baselines | blop, with abbot as a backend |
| Hierarchical or derived parameters, learning-curve early stopping, preference learning, SQL storage | Ax; not planned in abbot |

## Rules of thumb

- **Algorithms are written to travel**: policies and acquisitions stay plain
  BoTorch where possible, so an adapter or an upstream port is cheap.
- **The loop, its semantics and the UI are abbot's.**
- **Cover what a user would otherwise leave for**: where BoTorch has the
  numerics (standard, multi-objective, multi-fidelity and cost-aware BO), abbot
  supplies the spec, wiring and safe defaults, so a campaign never needs a second
  tool or a hand-rolled script.
- **Not ours to build**: heavy infrastructure with little BO in it: SQL storage
  and automatic model selection (Ax); hardware orchestration and device control
  (Bluesky, Xopt evaluators).
- **Adapters stay thin** and live in `abbot.interop`, one per neighbor, each
  behind an optional extra.
- **UX is a priority, and gets checked.** abbot aims for specs a scientist can
  read and safe behavior by default, and periodically sets its workflows up
  alongside the neighbors to find what's awkward in its own (see the README
  roadmap).

## Interop status

| Neighbor | Status |
|---|---|
| Xopt | `abbot.interop.xopt` (`abbot[xopt]`): abbot in Xopt's loop, and Xopt's generators in abbot's loop |
| blop | Planned, deferred until a beamline will use it: a `Campaign`-backed blop `Optimizer`. abbot requires BoTorch 0.18, the same line as blop and Ax |
| Ax | None planned; complementary rather than integrated |
| bluesky-adaptive | Not planned unless blop doesn't cover a need |
