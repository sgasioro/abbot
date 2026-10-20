# abbot in the ecosystem

Where abbot sits next to BoTorch, Ax, Xopt and Bluesky/blop, what it owns, and
what it deliberately leaves to them. Neighbour details were checked against
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

## Neighbours

**BoTorch.** A library, not a loop: no problem spec, candidate ids, queue,
resume or fault handling. abbot adds the campaign layer, the BAX family and
composite measurements on top.

**Ax.** The closest overlap. Its `Client` covers experiment setup, trials,
failure marking, early stopping, JSON/database snapshots and Pareto fronts, with
a large team behind it. For single- or multi-objective optimization of scalar
measurements, Ax is a good answer. abbot still covers standard BO (`SimpleBO`,
`ParetoBO`, and multi-fidelity as it lands) so a lab never has to leave it: the
overhead of switching tools pushes people to a hand-rolled BoTorch script, not
to Ax. Ax's model is
objectives and trials; BAX goals, posterior-sample algorithm execution and raw
observations do not fit it without fighting the framework.

**Xopt.** Accelerator- and beamline-driven. Problems are a VOCS, data is a
DataFrame of scalars, generators `generate(n)` from it. Its `BaxGenerator` is
InfoBAX only (expected information gain over `GridOptimize` /
`CurvatureGridOptimize`), without discrete variables. abbot's BAX is broader
(InfoBAX, MeanBAX, SwitchBAX; arbitrary goals; finite candidate sets; greedy
kriging-believer batches). Interop: `abbot.interop.xopt` maps spec and VOCS and
wraps a `Campaign` as a gest `Generator`, so an abbot problem runs under
`Xopt(generator=...)`.

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
- **Ax**: objective-and-trial centric, heavy, tightly pinned to a BoTorch
  release; BAX goals sit awkwardly in it.
- **blop**: Bluesky-only by design.

The BAX algorithms on their own are portable and could live elsewhere. What
needs abbot is owning the loop: raw observation → `evaluate` → outputs with the
observation kept; candidate ids, the outstanding queue, retry/halt, supervisors
and resume; and a UI that needs no EPICS, Bluesky or Badger.

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
- **Adapters stay thin** and live in `abbot.interop`, one per neighbour, each
  behind an optional extra.

## Interop status

| Neighbour | Status |
|---|---|
| Xopt | `abbot.interop.xopt` (`abbot[xopt]`) |
| blop | Planned: a `Campaign`-backed blop `Optimizer`. Needs abbot on BoTorch 0.18 (blop and Ax require 0.18.x; abbot is on 0.17) |
| Ax | None planned; complementary rather than integrated |
| bluesky-adaptive | Not planned unless blop doesn't cover a need |
