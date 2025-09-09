# VeloxQ-Style QUBO Solver (Reconstruction) — Python POC

A fast, **physics-inspired** optimizer for QUBO / Ising problems with verification, plotting, history export, and run logging — implemented entirely in Python.

> **Inspiration & Credit**
> This proof-of-concept (POC) is inspired by **VeloxQ 1**, a quantum-inspired classical optimizer announced by the Polish company **Quantumz.io**. VeloxQ’s public materials emphasize *classical hardware, extreme scale, topology-agnostic execution, and strong performance on QUBO/HUBO*.
> **This repository is an independent reconstruction**: it does **not** implement Quantumz.io’s proprietary algorithm, nor use their code or SDK. It is a faithful engineering POC in the same spirit—*continuous dynamics + heuristic control*—intended for research and teaching.
> **POC Author:** *Matthew Combatti* (this reconstruction, design, and code).

---

## Table of Contents

* [What this is](#what-this-is)
* [Why this exists](#why-this-exists)
* [Features](#features)
* [How it works (engineer’s gist)](#how-it-works-engineers-gist)
* [Math sketch](#math-sketch)
* [Install](#install)
* [Quick start](#quick-start)

  * [Built-in demo (MAX-CUT)](#built-in-demo-max-cut)
  * [Solve your own QUBO](#solve-your-own-qubo)
  * [Verify a solution](#verify-a-solution)
  * [Large-n demo + plot](#large-n-demo--plot)
* [CLI reference](#cli-reference)
* [API reference](#api-reference)
* [File formats](#file-formats)
* [Reproducibility & logging](#reproducibility--logging)
* [Performance notes](#performance-notes)
* [Security & correctness checks](#security--correctness-checks)
* [FAQ](#faq)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## What this is

* A **VeloxQ-style** optimizer for **QUBO** (Quadratic Unconstrained Binary Optimization):

  $\min_{x\in\{0,1\}^n} \; x^\top Q x \;+\; b^\top x$
* Implemented in a single Python module: `veloxq_reconstructed.py`.
* Packs in:

  * Continuous-state dynamics with momentum & damping
  * Binary double-well “attractor” that anneals the state toward {0,1}
  * Auto-tuned time step via spectral radius of $2Q$
  * Multi-restarts, plateau kicks, and greedy **1-opt polish**
  * **Verification** (exact enumeration via Gray code for small n, plus local-optimality certificate for any n)
  * **Plotting**, **history export (NPY/CSV)**, and **run JSON logs**
  * **MAX-CUT helper** utilities and demos

---

## Why this exists

Public info about **VeloxQ 1** highlights a modern, classical approach to large-scale QUBO/HUBO without annealer embedding constraints. Because the **exact update rules are proprietary**, this project reconstructs a practical, physics-inspired pipeline that *feels similar in spirit*:

* classical hardware
* deterministic dynamics with heuristic control
* scale via sparsity and parallelizable math
* built-in evaluation to keep solutions honest

This is a **research POC** you can study, extend, and compare against your favorite heuristics, CP/MIP solvers, and annealing-style methods.

---

## Features

* 🧠 **VeloxQ-style dynamics:** continuous states with inertial update, damping, and annealed double-well regularization to encourage binarization.
* 🧪 **Verification suite:**

  * **Exact** Gray-code enumerator for small $n$ (≈ up to 20–22 variables).
  * **Local-optimality** certificate (1-flip test) for any $n$.
* 📈 **Plotting & history:** energy trajectory charts (`--plot`), plus exports to `.npy` and `.csv`.
* 🧾 **Run logging:** JSON logs capturing config, final energy, wall time, dt, artifacts, and verification outcomes.
* 🧩 **MAX-CUT helpers:** generate weighted Erdos-Rényi graphs; convert MAX-CUT→QUBO.
* ⚙️ **Tunable & reproducible:** seeded RNG, restarts, step schedules, greedy polish.

---

## How it works (engineer’s gist)

1. You supply a QUBO $Q$ (symmetric) and optional $b$.
2. The solver runs **continuous-state dynamics** with velocity (momentum) and damping:

$$
\begin{aligned}
\dot{\mathbf v} &\leftarrow -\,\nabla E(\mathbf x)\;-\;\beta(t)\,\nabla R(\mathbf x)\;-\;\gamma\,\mathbf v \\
\mathbf x &\leftarrow \mathrm{clip}\!\big(\mathbf x + \Delta t \cdot \mathbf v,\,[0,1]\big)
\end{aligned}
$$



   where $E(x)=x^\top Q x + b^\top x$, and $R(x)$ is a **double-well** term that prefers values near 0 or 1.

3. $\beta(t)$ increases over time (anneal), coaxing the system toward binary corners.
4. We **sample**, **threshold**, and **retain the best** solution found; add **kicks** if progress plateaus; and do **greedy 1-opt** polishing at the end.
5. Small instances can be **exactly verified**; larger ones get a strong **local-opt** certificate.

---

## Math sketch

* **Energy** (continuous proxy): $E(x) = x^\top Q x + b^\top x$.
* **Binary regularizer:** $f(x)=\sum_i x_i^2(1-x_i)^2$ with gradient $\nabla f_i = 2x_i(1-x_i)(1-2x_i)$.
* **Total force:** $-\nabla E(x) \;-\; \beta(t)\nabla f(x)$.
* **Damped inertial update:** velocity with damping $\gamma$ and optional noise that decays over time.
* **Auto step size:** $\Delta t$ selected from the spectral radius of $2Q$ for stability and speed.

> This is a **reconstruction** using well-known physics-inspired optimization ingredients in the same spirit as the claims about VeloxQ; it is not the proprietary algorithm.

---

## Install

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy matplotlib
```

> `matplotlib` is only needed if you use `--plot`.

Place the main file in your working directory:

* `veloxq_reconstructed.py` (the solver and CLI)

Optional example scripts (nice to have, but not required):

* `run_example.py` (6-var QUBO end-to-end demo)
* `run_large_demo.py` (n=200 MAX-CUT demo with plot and artifacts)

---

## Quick start

### Built-in demo (MAX-CUT)

Generate a random weighted graph, convert to QUBO, solve, verify, and plot:

```bash
python veloxq_reconstructed.py demo_maxcut --n 60 --p 0.2 --seed 7 \
  --steps 5000 --restarts 3 \
  --plot energy.png \
  --save-history-npy hist.npy \
  --save-history-csv hist.csv \
  --save-run-json run.json \
  --exact
```

You’ll see:

* Final energy and (negative) cut equivalence
* 1-opt local-optimality certificate
* Exact optimum (since you chose `--exact` and n≤22; for n=60 exact is skipped)
* `energy.png`, `hist.npy`, `hist.csv`, and `run.json` artifacts

### Solve your own QUBO

Create an `.npz` file containing `Q` (and optional `b`):

```python
import numpy as np
from veloxq_reconstructed import save_qubo_npz

# Example Q, b
rng = np.random.default_rng(0)
n = 24
Q = rng.standard_normal((n, n))
Q = 0.5*(Q + Q.T)  # symmetrize
b = rng.standard_normal(n)

save_qubo_npz("myqubo.npz", Q, b)
print("Saved myqubo.npz")
```

Solve it:

```bash
python veloxq_reconstructed.py solve_qubo --in myqubo.npz \
  --steps 6000 --restarts 5 \
  --plot energy.png \
  --save-history-npy hist.npy \
  --save-history-csv hist.csv \
  --save-run-json run.json \
  --out sol.npy \
  --exact
```

### Verify a solution

```bash
python veloxq_reconstructed.py verify --in myqubo.npz --x sol.npy --exact
```

* If $n \le 22$: runs **exact** enumeration (Gray code) and reports the gap.
* Always reports 1-opt local-optimality.

### Large-n demo + plot

A turnkey n=200 MAX-CUT demo (writes a QUBO, solves, saves artifacts, and plots).

```bash
python run_large_demo.py
```

Artifacts:

* `large_qubo_200.npz`, `sol_200.npy`
* `energy_history.(npy|csv)`, `energy_history.png`
* Console logs with final energy and 1-opt certificate

---

## CLI reference

All flags work for both `demo_maxcut` and `solve_qubo` (unless noted).

| Flag                          | Meaning                                        |
| ----------------------------- | ---------------------------------------------- |
| `--steps INT`                 | Number of integration steps (default 5000)     |
| `--restarts INT`              | Independent restarts (default 3)               |
| `--sample-every INT`          | Record energy every k steps (default 10)       |
| `--beta0 FLOAT`               | Initial regularizer strength (default 0.1)     |
| `--beta-max FLOAT`            | Final regularizer strength (default 8.0)       |
| `--noise0 FLOAT`              | Initial velocity noise (anneals to 0)          |
| `--gamma FLOAT`               | Velocity damping (default 0.2)                 |
| `--dt FLOAT`                  | Override auto-tuned step size (optional)       |
| `--plateau INT`               | Steps without improvement before a “kick”      |
| `--polish INT`                | Greedy 1-opt iterations at end                 |
| `--seed INT`                  | RNG seed                                       |
| `--plot PATH.png`             | Save energy trajectory plot                    |
| `--save-history-npy PATH.npy` | Save energy history (NumPy)                    |
| `--save-history-csv PATH.csv` | Save energy history (CSV)                      |
| `--save-run-json PATH.json`   | Save full run log (config, results, artifacts) |
| `--exact`                     | If feasible (n≤22), run exact verification     |
| `--nmax INT`                  | Max n for exact enumeration (default 22)       |

**Commands:**

* `demo_maxcut` — synthesize graph $G(n, p)$, convert to QUBO, solve.

  * `--n INT`, `--p FLOAT`, `--seed INT`

* `solve_qubo` — solve an existing QUBO `.npz`.

  * `--in PATH.npz`, optional `--out sol.npy`

* `verify` — evaluate a saved solution.

  * `--in PATH.npz`, `--x sol.npy` (no plotting/history flags here)

---

## API reference

From `veloxq_reconstructed.py`:

```python
from veloxq_reconstructed import (
    Qubo,
    VXQConfig,
    VeloxQReconstructed,
    # Helpers
    maxcut_to_qubo,
    random_maxcut_weights,
    save_qubo_npz, load_qubo_npz,
    # Verification
    exact_min_qubo_gray, local_optimality_certificate,
)
```

* `Qubo.from_arrays(Q, b=None)` → QUBO object (symmetrizes `Q`).

* `Qubo.energy(x_bin)` → scalar energy for binary `x`.

* `Qubo.delta_flip(x_bin)` → per-bit ΔE for flipping each bit.

* `VXQConfig(...)` → configuration dataclass (all CLI flags map 1:1).

* `VeloxQReconstructed(qubo, cfg).solve()` → `(x_bin, energy, meta)`
  `meta` contains `history` (energies at sample points) and `dt`.

* `maxcut_to_qubo(W)` → derive QUBO from weighted adjacency `W`.

* `random_maxcut_weights(n,p,seed)` → weighted Erdos–Rényi graph.

* `save_qubo_npz(path, Q, b)` / `load_qubo_npz(path)` → IO helpers.

* `exact_min_qubo_gray(Q, b, nmax)` → `(x*, E*)` or `None`.

* `local_optimality_certificate(qubo, x)` → min 1-flip ΔE (≥0 ⇒ 1-opt).

---

## File formats

* **QUBO problem** (`.npz`):

  * `Q`: `(n, n)` float64 symmetric
  * `b`: `(n,)` float64 (optional)
* **Solution** (`.npy`): `(n,)` `uint8` vector in `{0,1}`.
* **History** (`.npy` / `.csv`): sequence of energies at sample points.
* **Run log** (`.json`): config, timings, energies, artifacts, verification.

---

## Reproducibility & logging

* Set `--seed` to lock in RNG behavior (initial conditions, kicks).
* The solver auto-tunes `dt` from the spectral radius of `2Q`; you can override `--dt` for experiments.
* Use `--save-run-json run.json` to capture:

  * problem size, config, final energy, wall time, dt
  * 1-opt certificate and exact verification (when applicable)
  * artifact paths (plot/history/solution)

---

## Performance notes

* **Exact verification** is exponential; keep `n ≤ 22`. For larger problems, rely on the **1-opt certificate** and empirical robustness (restarts + polish).
* Increase `--steps` and `--restarts` for tougher instances.
* For dense large `Q`, NumPy BLAS dominates time; for sparse problems, consider switching to a sparse representation (future extension).
* The annealing profile (β schedule), plateau kicks, and polish iterations are impactful; tune to taste.

---

## Security & correctness checks

* Energy sanity: for a given `x`, `Qubo.energy(x)` is deterministic.
* Local optimality: `local_optimality_certificate` reports the most negative ΔE; non-negative ⇒ no single bit flip can improve energy.
* Exactness: `exact_min_qubo_gray` confirms global optimality on small problems and reports the **gap** for your heuristic solution.

---

## FAQ

**Q: Is this the real VeloxQ algorithm?**
A: No. This is an **independent POC reconstruction** inspired by publicly stated characteristics of VeloxQ 1 (classical, physics-inspired QUBO solver). It does not implement proprietary internals.

**Q: What problems can I model as QUBO?**
A: MAX-CUT, max-clique, k-partition, portfolio selection, knapsack, scheduling/routing variants, and many ML regularization tasks — if you can cast it to quadratic binary form.

**Q: How do I handle HUBO (higher-order)?**
A: Standard technique: reduce HUBO→QUBO with auxiliary variables (not included here to keep the POC minimal). Once reduced, solve as usual.

**Q: How do I know if my result is “good”?**
A: For small n, run `--exact`. For larger n, report energy and pass the 1-opt certificate (≥0). Compare against other heuristics or MIP relaxations for additional confidence.

---

## License

**MIT License** — do whatever you’d like, just keep the attribution.

* POC reconstruction author: **Matthew Combatti**
* Inspiration credit: **Quantumz.io (Poland)** for VeloxQ 1’s publicly described approach and benchmarks that motivated this classical, educational reconstruction.

---

## Acknowledgments

* **Quantumz.io** (Poland) — for the **VeloxQ 1** concept and public performance claims that inspired building an open, educational reconstruction on classical hardware.
* The broader community of **physics-inspired optimization**: simulated bifurcation/annealing, damped inertial dynamics, double-well regularizers, and deterministic annealing ideas that inform this POC design.

---

### Badge of clarity

> ⚠️ **Disclaimer:** This repository is **not affiliated with or endorsed by Quantumz.io**. All trademarks and product names are the property of their respective owners. This is an **educational POC**, reconstructed from first principles and public high-level descriptions, intended for research, experimentation, and teaching.
