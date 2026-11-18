"""CPU vs. CUDA wall time for one model fit and one proposal (``Campaign(device=)``).

For each number of runs ``n`` and inputs ``d``, a campaign is given ``n`` runs of a
smooth noisy function and timed through the public path: one ``current_model()``
(the fit) and one ``suggest()`` per policy on that fit. Proposals include moving
designs to the device and back; BAX's candidate grid is built beforehand (a
campaign builds it once).

Each (device, n, d) runs in a fresh process, after a small warm-up fit and proposal
there (so CUDA's start-up isn't counted, and memory doesn't carry over). A process
that is killed (out of memory) or runs past ``--timeout`` is recorded as such, and
the policies it hadn't reached are retried in a new process.

Policies:
  SimpleBO       noisy log-EI, optimized from 10 restarts
  InfoBAX G/S    BAXPolicy(TopK(k=10), "InfoBAX") on a grid of G candidates with
                 S posterior samples

    python benchmarks/gpu_timing.py --n 50 200 1000 3000 --d 3 8 --out gpu_timing.jsonl
    python benchmarks/gpu_timing.py --show gpu_timing.jsonl      # the table again

Results go to ``--out`` (JSON lines, one per device, n, d and policy; appended, so
runs split across invocations, e.g. ``--devices cpu`` and ``--devices cuda``, build
one file) and a table is printed: seconds for fit + proposal on each device, and the
CUDA speedup. Run at most a few of these at once (memory), one GPU each.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import warnings
from collections import defaultdict

import torch

POLICIES = ("SimpleBO", "InfoBAX 1k/20", "InfoBAX 1k/100", "InfoBAX 10k/20", "InfoBAX 10k/100")


def _policy(name: str):
    from abbot import BAXPolicy, SimpleBO, TopK

    if name == "SimpleBO":
        return SimpleBO()
    grid, samples = name.split()[1].split("/")
    return BAXPolicy(TopK(k=10, output="f"), "InfoBAX",
                     n_candidates=int(grid.replace("k", "000")),
                     n_posterior_samples=int(samples))


def _spec(d: int):
    from abbot import ContinuousDomain, OutputSpec, ProblemSpec

    return ProblemSpec(
        domain=[ContinuousDomain(name=f"x{i}", lower=0.0, upper=1.0) for i in range(d)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )


def _truth(X: torch.Tensor) -> torch.Tensor:
    """A smooth bump with ripples, best near x = 0.3 in every input."""
    return -((X - 0.3) ** 2).sum(-1) + 0.1 * torch.sin(6 * X).sum(-1)


def _history(spec, n: int, seed: int):
    from abbot import History, Record

    X = spec.sample_designs(n, seed=seed)
    g = torch.Generator().manual_seed(seed)
    Y = _truth(X) + 0.02 * torch.randn(n, generator=g, dtype=torch.double)
    return History([Record(candidate=x, observation=None, outputs=y.reshape(1)) for x, y in zip(X, Y)])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_one(n: int, d: int, device: torch.device, policies, seed: int = 0):
    """Fit once on ``n`` runs, then one proposal per policy on that fit; yields a row
    per policy as it finishes."""
    from abbot import Campaign, SimpleBO

    spec = _spec(d)
    c = Campaign(spec, SimpleBO(), device=device, initial_designs=torch.empty(0, d),
                 on_model_error="raise")
    c.load_history(_history(spec, n, seed))
    torch.manual_seed(seed)
    start = time.perf_counter()
    c.current_model()
    _sync(device)
    fit = time.perf_counter() - start
    for name in policies:
        c.policy = _policy(name)
        if hasattr(c.policy, "candidate_grid"):
            c.policy.candidate_grid(spec)  # built once per campaign, on the CPU
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        (candidate,) = c.suggest(1)
        _sync(device)
        propose = time.perf_counter() - start
        c.cancel(candidate.id)
        yield {
            "device": device.type, "n": n, "d": d, "policy": name, "seed": seed,
            "fit_s": fit, "propose_s": propose, "total_s": fit + propose,
            "peak_gpu_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                            if device.type == "cuda" else None),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }


def _machine(device: torch.device) -> dict:
    info = {"torch": torch.__version__, "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _child(device: str, n: int, d: int, policies: list[str], seed: int) -> None:
    """One (device, n, d) in this process: a warm-up, then a JSON row per policy
    on stdout."""
    warnings.filterwarnings("ignore")
    device = torch.device(device)
    for _ in time_one(20, 2, device, ["SimpleBO", "InfoBAX 1k/20"]):  # warm-up
        pass
    machine = _machine(device)
    for row in time_one(n, d, device, policies, seed):
        print(json.dumps({**row, **machine}), flush=True)


def _run(device: str, n: int, d: int, policies: list[str], seed: int, timeout: float) -> list[dict]:
    """``_child`` in a new process, retrying the policies after one that killed it."""
    rows: list[dict] = []
    todo = list(policies)
    while todo:
        cmd = [sys.executable, __file__, "--child", device, str(n), str(d), str(seed), *todo]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            out, error = proc.stdout, None if proc.returncode == 0 else (
                "out of memory (killed)" if proc.returncode in (-9, 137) else
                (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1])
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            error = f"timeout ({timeout:.0f} s)"
        done = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        rows += done
        todo = todo[len(done):]
        if todo and error:
            rows.append({"device": device, "n": n, "d": d, "policy": todo[0], "seed": seed,
                         "error": error})
            todo = todo[1:]
    return rows


def show(rows: list[dict]) -> str:
    """A table: per (n, d), each policy's fit + proposal seconds on the CPU and on
    CUDA, and the speedup (CPU / CUDA; above 1 means CUDA is faster). Then the
    crossover: the smallest n at which CUDA wins, per d and policy."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["n"], r["d"], r["policy"])][r["device"]] = r
    policies = [p for p in POLICIES if any(k[2] == p for k in by)]
    sizes = sorted({(k[0], k[1]) for k in by}, key=lambda t: (t[1], t[0]))
    lines = ["Seconds for one fit + one proposal, CPU / CUDA (speedup = CPU / CUDA).", ""]
    lines.append("| d | n | fit only | " + " | ".join(policies) + " |")
    lines.append("|" + "---|" * (3 + len(policies)))
    for n, d in sizes:
        fits = [by[(n, d, p)] for p in policies if (n, d, p) in by]
        fit = {dev: next((v[dev]["fit_s"] for v in fits if "fit_s" in v.get(dev, {})), None)
               for dev in ("cpu", "cuda")}
        cells = [str(d), str(n), _pair(fit["cpu"], fit["cuda"])]
        for p in policies:
            cpu, gpu = (by.get((n, d, p), {}).get(dev, {}) for dev in ("cpu", "cuda"))
            text = _pair(_cell(cpu), _cell(gpu))
            if "total_s" in cpu and "total_s" in gpu:
                text += f" ({cpu['total_s'] / gpu['total_s']:.2g}×)"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Crossover (smallest n where CUDA's fit + proposal is faster):"]
    for d in sorted({k[1] for k in by}):
        for p in policies:
            ns = sorted(n for (n, dd, pp), v in by.items() if dd == d and pp == p
                        and "total_s" in v.get("cuda", {}) and v.get("cpu"))
            wins = [n for n in ns if "total_s" not in by[(n, d, p)]["cpu"]  # CPU failed
                    or by[(n, d, p)]["cuda"]["total_s"] < by[(n, d, p)]["cpu"]["total_s"]]
            where = f"n = {wins[0]}" if wins else f"none up to n = {ns[-1]}" if ns else "not run"
            lines.append(f"  d = {d}, {p}: {where}")
    machines = {(r.get("cpu"), r.get("threads"), r.get("gpu"), r.get("torch")) for r in rows if "torch" in r}
    for cpu, threads, gpu, version in sorted(machines, key=str):
        lines.append(f"\n{'GPU ' + gpu if gpu else 'CPU ' + str(cpu)}, {threads} threads, torch {version}")
    return "\n".join(lines)


def _cell(row: dict):
    if "total_s" in row:
        return row["total_s"]
    return "OOM" if "memory" in row.get("error", "") else "timeout" if "timeout" in row.get("error", "") else (
        "error" if row else None)


def _pair(cpu, gpu) -> str:
    def f(v):
        return "–" if v is None else v if isinstance(v, str) else f"{v:.3g}"

    return f"{f(cpu)} / {f(gpu)}"


def _best(rows: list[dict]) -> list[dict]:
    """Each configuration's fastest row (with repeats); a failure only if all failed."""
    best = {}
    for r in rows:
        key = (r["device"], r["n"], r["d"], r["policy"])
        old = best.get(key)
        if old is None or ("total_s" in r and ("total_s" not in old or r["total_s"] < old["total_s"])):
            best[key] = r
    return list(best.values())


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        device, n, d, seed, *policies = sys.argv[2:]
        _child(device, int(n), int(d), policies, int(seed))
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=[50, 200, 1000, 3000])
    parser.add_argument("--d", type=int, nargs="+", default=[3, 8])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--repeats", type=int, default=1, help="timed repeats (seeds) per configuration")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    parser.add_argument("--out", default="gpu_timing.jsonl")
    parser.add_argument("--show", metavar="FILE", help="print the table from FILE and exit")
    args = parser.parse_args()
    if args.show:
        with open(args.show) as f:
            print(show(_best([json.loads(line) for line in f if line.strip()])))
        return
    rows = []
    for d in args.d:
        for n in args.n:
            for device in args.devices:
                for seed in range(args.repeats):
                    for row in _run(device, n, d, args.policies, seed, args.timeout):
                        rows.append(row)
                        with open(args.out, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        took = (f"fit {row['fit_s']:8.3f}s propose {row['propose_s']:8.3f}s"
                                if "total_s" in row else row["error"])
                        print(f"{device:4s} d={d} n={n:5d} {row['policy']:15s} {took}", flush=True)
    print()
    print(show(_best(rows)))


if __name__ == "__main__":
    main()
