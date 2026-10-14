# abbot examples

Runnable Jupyter notebooks, each a self-contained story covering one of abbot's
capabilities. They use tiny budgets so they execute in seconds.

| Notebook | Shows |
|---|---|
| `01_lab_bo.ipynb` | Single-objective lab BO over a **mixed continuous + categorical** design space — the non-expert entry point. |
| `02_multi_objective.ipynb` | **Native multi-objective** optimization with `ParetoBO`; recovers the Pareto front. |
| `03_composite_sbi.ipynb` | **Composite simulation-based inference** — objective = a functional of an inferred posterior, with cached-posterior re-scoring. abbot's lead differentiator. |
| `04_goal_bax.ipynb` | **Goal-driven search (BAX)** — find the *region* satisfying a target, not just the maximum. |
| `custom_models.py` | **Swapping the surrogate** — a Matern-kernel GP, a fully-Bayesian SAAS GP, and a deep ensemble, all behind the same `Campaign` via the `ModelConstructor` seam (run `python examples/custom_models.py`). |

## Run

```bash
pip install -e '.[examples]'       # jupyter + matplotlib
jupyter lab examples/              # or: jupyter notebook

# regenerate the notebooks from source (they are built, not hand-edited):
python examples/_build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace examples/[0-9]*.ipynb
```

The notebooks are smoke-tested (`tests/test_examples.py`, in the `slow` tier) so
they can't drift out of sync with the API.
"""
