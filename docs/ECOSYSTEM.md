# abbot in the ecosystem

Where abbot sits next to BoTorch, Ax, Xopt, Bluesky/blop and other lab-oriented
tools, what it owns, and what it deliberately leaves to them. Neighbor details
were checked against ax-platform 1.3.1, xopt 3.2.2 and blop 1.1.0 (October 2026;
the Xopt details were re-checked against its main branch in November 2026); the
other lab-oriented tools are summarized from their papers and package pages
(November 2026). They move, so re-check before relying on a specific claim.

## Position

abbot is a standalone decision engine for experiments whose **goal isn't an
optimum** (BAX goals: a level set, the best point per band, every condition in
spec) or whose **measurement isn't a number** (a spectrum, an SBI posterior,
replicates reduced by an `evaluate`). It runs a supervised ask/tell loop that
survives flaky experiments, and ships a lab UI light enough to start a campaign
in minutes.

abbot grew out of a need that was being met by several scattered,
problem-specific packages. Labs work in different modes (a control room, a
beamline, a bench), and neighboring tools already serve some of them well, so the
emphasis as abbot develops is on what can be shared with them and what is better
kept separate. It follows the gest-api generator standard and connects to the
orchestrators people already use through thin adapters in `abbot.interop`.
Algorithms that fit another project naturally may be offered upstream.

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
DataFrame of scalars, and generators follow the gest-api `suggest(n)` / `ingest`
protocol. Its `BaxGenerator` is
InfoBAX only (expected information gain over `GridOptimize` /
`CurvatureGridOptimize`), without discrete variables, and the
[`bax-algorithms`](https://pypi.org/project/bax-algorithms/) package (xopt-org)
adds further BAX algorithms for Xopt generators. abbot's BAX covers InfoBAX,
MeanBAX and SwitchBAX, arbitrary goals, finite candidate sets and greedy
kriging-believer batches. Interop runs both ways over the gest-api
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

**Other lab-oriented BO tools.** Several packages bring BO to experimental
science, each with its own emphasis. None of them is integrated with abbot;
they are listed so a reader can pick the right tool.
- [**Atlas**](https://github.com/aspuru-guzik-group/atlas) (Aspuru-Guzik group;
  Hickman et al., *Digital Discovery* 2025): BO as the planner for self-driving
  labs, used with the group's Olympus and ChemOS 2.0. It covers mixed and
  molecular spaces, known constraints (any function of the inputs, simplex
  compositions), unknown constraints learned from failed runs, multi-objective
  scalarizations (Chimera and others), robust optimization, parameters that are
  best on average across substrates, multi-fidelity, meta-learning from earlier
  campaigns, and asynchronous runs. The closest overlap with abbot is learning
  where runs fail: Atlas treats a failed measurement as a property of the design,
  while abbot separates apparatus faults (retried, then resolved by a person)
  from bad regions (learned and avoided).
- [**BoFire**](https://github.com/experimental-design/bofire) (Dürholt et al.,
  *JMLR* 2025): BO and classical design of experiments for chemistry, with
  continuous, discrete, categorical and molecular inputs, close-to-target
  objectives, multi-objective strategies and pydantic serialization.
- [**BayBE**](https://github.com/emdgroup/baybe) (*Digital Discovery* 2025): a Bayesian back end for experimental
  planning with little or no data, including transfer learning from related
  campaigns.
- [**gpCAM**](https://github.com/lbl-camera/gpCAM) (Berkeley Lab): Gaussian
  process modeling and autonomous data acquisition, with user-supplied
  acquisition and kernel functions and HPC-scale optimization.
- **multibax** (Chitturi et al., *npj Computational Materials* 2024): the
  reference implementation of target-subset BAX (MeanBAX, InfoBAX, SwitchBAX)
  that abbot's goals and BAX policies port.

abbot's emphasis among them is goal-driven search (BAX) next to optimization,
measurements that aren't a single number, and campaigns a person supervises.

## Why not extend one of them

Several tools in this space coexist, each shaped by its users. abbot's algorithms
would mostly fit elsewhere; what doesn't fit is the campaign layer around them,
which in each neighbor is shared infrastructure that its other pieces rely on.

**Xopt.** A new algorithm is a new generator. What abbot needs would instead
change the parts every generator, and Badger, sit on:
- **The data model.** Xopt's data is a DataFrame of scalars. abbot keeps each
  run's raw observation (a spectrum, posterior samples), caches its analysis, and
  can re-score the whole history when the target changes (`Rescore`).
- **The spec.** Sources with a target in physical units, labeled categories,
  log-scaled variables, outputs that are observed but not optimized, named output
  constraints and per-variable travel limits. Xopt's spec, VOCS, is now the
  gest-api standard shared across projects, so extending it is a standards
  question rather than a single project's change.
- **The loop.** Candidate ids, an outstanding queue and a retry queue; halts that
  a person resolves (retry, exclude, bad region) and that survive restarts; a
  verdict on every run. Xopt's loop stops on an error or skips the run
  (`strict`).
- **Behavior across all policies.** The usability classifier, feasibility
  weighting, travel limits and pending designs apply under every policy, BAX
  included. In Xopt each generator owns its acquisition, so the same behavior
  would be repeated in each one.
- **The hand-over.** A recommendation at the target, with whether it was run
  there and its chance of being usable and feasible.

Its users are also different: Badger is a control-room tool over control
systems; abbot's UI is for a scientist entering measurements at a bench.

**Ax.** Built around objectives and trials, which BAX goals and raw composite
observations don't map onto.

**blop.** Bluesky-only by design.

**What could travel.** Pieces that fit a neighbor as an algorithm or a model, and
could be offered upstream where wanted: the BAX strategies and arbitrary goals
(as Xopt BAX algorithms), multi-fidelity entropy search, and the prior-mean and
correlated-output model constructors. They would arrive without abbot's spec,
so physical units, labels and sources would be translated by hand there. Interop
keeps the tools connected: abbot runs as an Xopt generator and uses Xopt's
generators through gest-api.

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
