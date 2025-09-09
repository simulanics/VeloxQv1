#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VeloxQ-style QUBO Solver (Reconstruction) + Evaluator/Verifier + Plotting + History Export + Run Log
====================================================================================================
- Physics-inspired continuous dynamics with momentum & damping
- Binary double-well regularizer annealed to {0,1}
- Auto-tuned step size via spectral radius
- Restarts, plateau kicks, greedy 1-opt polish
- Exact Gray-code verifier for small N
- MAX-CUT helpers
- Plot energy history with --plot <path.png>
- Export energy history: --save-history-npy <file.npy> / --save-history-csv <file.csv>
- NEW: Save a run log JSON: --save-run-json <file.json>

Usage examples
--------------
# Demo on random MAX-CUT (with plot + history + run log)
python veloxq_reconstructed.py demo_maxcut --n 60 --p 0.2 --seed 7 \
  --plot energy.png --save-history-npy hist.npy --save-history-csv hist.csv \
  --save-run-json run.json

# Solve a saved QUBO and export artifacts + run log
python veloxq_reconstructed.py solve_qubo --in example_qubo.npz --steps 4000 --restarts 5 \
  --plot energy.png --save-history-npy hist.npy --save-history-csv hist.csv \
  --out sol.npy --save-run-json run.json

# Verify solution
python veloxq_reconstructed.py verify --in example_qubo.npz --x sol.npy --exact
"""

from __future__ import annotations
import argparse
import sys
import time
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


# ---------------------------
# QUBO utilities & structures
# ---------------------------

@dataclass
class Qubo:
    Q: np.ndarray           # symmetric (n,n)
    b: Optional[np.ndarray] # (n,) or None

    @staticmethod
    def from_arrays(Q: np.ndarray, b: Optional[np.ndarray] = None) -> "Qubo":
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square (n x n).")
        # Symmetrize for safety (QUBO convention)
        Q = 0.5 * (Q + Q.T)
        n = Q.shape[0]
        if b is None:
            b = None
        else:
            b = np.asarray(b, dtype=float).reshape(n)
        return Qubo(Q=Q, b=b)

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, x_bin: np.ndarray) -> float:
        """QUBO energy for a BINARY vector x ∈ {0,1}^n."""
        x = x_bin.astype(float, copy=False)
        val = float(x @ self.Q @ x)
        if self.b is not None:
            val += float(self.b @ x)
        return val

    def grad(self, x_cont: np.ndarray) -> np.ndarray:
        """Gradient ∇(x^T Q x + b^T x) for continuous x in [0,1]^n."""
        g = 2.0 * (self.Q @ x_cont)
        if self.b is not None:
            g = g + self.b
        return g

    def delta_flip(self, x_bin: np.ndarray) -> np.ndarray:
        """
        Energy change ΔE for flipping each bit of binary x:
          ΔE_i = 2*d*(Qx)_i + Q_ii + d*b_i,  d = 1-2*x_i ∈ {+1,-1}
        """
        Qx = self.Q @ x_bin
        d = 1 - 2 * x_bin  # +1 if 0->1, -1 if 1->0
        diag = np.diag(self.Q)
        if self.b is None:
            return 2.0 * d * Qx + diag
        else:
            return 2.0 * d * Qx + diag + d * self.b


# ---------------------------
# Auto-tuning / spectral norm
# ---------------------------

def spectral_radius_power_iteration(M: np.ndarray, iters: int = 64, seed: int = 0) -> float:
    """Return |Rayleigh quotient| ≈ spectral radius for symmetric M."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    v = rng.normal(size=n)
    v /= (np.linalg.norm(v) + 1e-12)
    lam = 0.0
    for _ in range(iters):
        v = M @ v
        nrm = float(np.linalg.norm(v))
        if nrm == 0.0:
            return 0.0
        v /= nrm
        lam = float(v @ (M @ v))
    return abs(lam)


# ---------------------------
# VeloxQ-style solver (recon)
# ---------------------------

@dataclass
class VXQConfig:
    steps: int = 5000
    restarts: int = 3
    sample_every: int = 10
    beta0: float = 0.1      # strength of binary double-well at start
    beta_max: float = 8.0   # strength at end of schedule
    noise0: float = 0.01    # initial velocity noise (anneals to 0)
    gamma: float = 0.2      # velocity damping
    dt: Optional[float] = None  # time step; auto if None
    plateau_patience: int = 200 # steps since last improvement before a "kick"
    polish_iters: int = 2000    # greedy 1-opt flips at end
    seed: int = 42
    verbose: bool = True


class VeloxQReconstructed:
    """
    Continuous dynamics + inertial/momentum + binary regularizer.
    The regularizer uses f(x)=x^2(1-x)^2 with ∇f = 2x(1-x)(1-2x), annealed by beta(t).
    """
    def __init__(self, qubo: Qubo, cfg: VXQConfig):
        self.qubo = qubo
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _autotune_dt(self) -> float:
        # Use spectral radius of Hessian ~ 2Q to pick a stable step.
        rho = spectral_radius_power_iteration(2.0 * self.qubo.Q, iters=64, seed=self.cfg.seed)
        return 0.9 / (rho + 1e-12) if rho > 0 else 0.5

    def solve(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        n = self.qubo.n
        cfg = self.cfg

        dt = cfg.dt if cfg.dt is not None else self._autotune_dt()
        if cfg.verbose:
            print(f"[VXQ] n={n} | steps={cfg.steps} | restarts={cfg.restarts} | dt={dt:.4g}")

        best_xb = None
        best_E = float("inf")
        history = []

        for r in range(cfg.restarts):
            # random initialization in (0,1)
            x = self.rng.random(n)
            v = np.zeros(n)
            beta = cfg.beta0
            sigma = cfg.noise0
            last_best_step = -1

            for t in range(cfg.steps):
                g = self.qubo.grad(x)  # gradient of QUBO energy
                reg = 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x)   # ∇[x^2(1-x)^2]
                force = -g - beta * reg

                v = (1.0 - cfg.gamma) * v + dt * force
                if sigma > 0.0:
                    v += sigma * self.rng.normal(size=n)
                x = x + dt * v
                np.clip(x, 0.0, 1.0, out=x)

                # linear schedules
                frac = t / max(1, cfg.steps - 1)
                beta = min(cfg.beta_max, cfg.beta0 + (cfg.beta_max - cfg.beta0) * frac)
                sigma = cfg.noise0 * (1.0 - frac)

                # periodic sampling
                if (t % cfg.sample_every) == 0 or t == (cfg.steps - 1):
                    xb = (x >= 0.5).astype(int)
                    E = self.qubo.energy(xb)
                    history.append(E)
                    if E < best_E - 1e-12:
                        best_E, best_xb, last_best_step = E, xb.copy(), t

                # plateau kick to escape flat regions
                if last_best_step >= 0 and (t - last_best_step) >= cfg.plateau_patience:
                    kick = 0.1 * self.rng.standard_normal(n)
                    x[:] = np.clip(x + kick, 0.0, 1.0)
                    last_best_step = t

            if cfg.verbose:
                print(f"[VXQ] restart {r+1}/{cfg.restarts}: incumbent E={best_E:.6f}")

        # Greedy 1-opt polishing (guarantees local optimality wrt single flips)
        x = best_xb.copy()
        for _ in range(cfg.polish_iters):
            deltas = self.qubo.delta_flip(x)
            i = int(np.argmin(deltas))
            if deltas[i] < -1e-12:
                x[i] = 1 - x[i]
            else:
                break
        final_E = self.qubo.energy(x)
        if cfg.verbose:
            print(f"[VXQ] polish: Δ={best_E - final_E:+.6f} → final E={final_E:.6f}")

        meta: Dict[str, Any] = {"history": np.array(history, dtype=float), "dt": float(dt)}
        return x, final_E, meta


# ---------------------------
# Exact verification (Gray code)
# ---------------------------

def exact_min_qubo_gray(Q: np.ndarray, b: Optional[np.ndarray], nmax: int = 22) -> Optional[Tuple[np.ndarray, float]]:
    """
    Exact solver for min QUBO via Gray-code enumeration with O(n) incremental updates.
    Safe default up to n≈20..22 (2^n states). Returns (x*, E*) or None if n>nmax.
    """
    n = Q.shape[0]
    if n > nmax:
        return None

    if b is None:
        b = np.zeros(n, dtype=float)

    # Start from x=0^n
    x = np.zeros(n, dtype=int)
    E = 0.0
    Qx = np.zeros(n, dtype=float)

    best_E = E
    best_x = x.copy()

    total = 1 << n
    for k in range(1, total):
        # bit index that flips in Gray code is ctz(k)
        i = (k & -k).bit_length() - 1
        d = 1 - 2 * x[i]  # +1 if 0→1, -1 if 1→0

        # ΔE = 2 d (Qx)_i + Q_ii + d b_i   using current Qx before flip
        dE = 2.0 * d * Qx[i] + Q[i, i] + d * b[i]
        E += dE

        # update state
        x[i] = 1 - x[i]
        Qx += d * Q[:, i]

        if E < best_E:
            best_E, best_x = E, x.copy()

    return best_x, float(best_E)


def local_optimality_certificate(qubo: Qubo, x_bin: np.ndarray) -> float:
    """
    Return the most negative single-bit ΔE (should be >= 0 at strict local optimum).
    """
    deltas = qubo.delta_flip(x_bin)
    return float(np.min(deltas))


# ---------------------------
# MAX-CUT → QUBO helpers
# ---------------------------

def maxcut_to_qubo(W: np.ndarray) -> Qubo:
    """
    Given symmetric nonnegative weights W (diag=0), MAXIMIZE cut weight.
    We MINIMIZE energy => set Q,b so that E(x) = x^T Q x + b^T x = - Cut(x) + const.
    Derivation yields:
      Q_ij =  W_ij  for i!=j
      b_i  = -sum_{j≠i} W_ij
      Q_ii =  0
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")
    n = W.shape[0]
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    b = -np.sum(W, axis=1)
    Q = W.copy()
    np.fill_diagonal(Q, 0.0)
    return Qubo.from_arrays(Q, b)


def random_maxcut_weights(n: int, p: float, seed: int = 0) -> np.ndarray:
    """
    Erdos–Renyi graph with edge prob p, random weights in (0,1).
    """
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(float)
    W = np.triu(A * rng.random((n, n)), 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


# ---------------------------
# I/O helpers
# ---------------------------

def save_qubo_npz(path: str, Q: np.ndarray, b: Optional[np.ndarray]) -> None:
    if b is None:
        np.savez_compressed(path, Q=np.asarray(Q, dtype=float))
    else:
        np.savez_compressed(path, Q=np.asarray(Q, dtype=float), b=np.asarray(b, dtype=float))


def load_qubo_npz(path: str) -> Qubo:
    data = np.load(path, allow_pickle=False)
    Q = data["Q"]
    b = data["b"] if "b" in data.files else None
    return Qubo.from_arrays(Q, b)


# ---------------------------
# Plotting + History export + Run log helpers
# ---------------------------

def maybe_plot_history(history: np.ndarray, plot_path: Optional[str]) -> None:
    if plot_path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            f"--plot was provided but matplotlib is not available. "
            f"Install it with `pip install matplotlib`. Original error: {e}"
        )
    hist = np.asarray(history, dtype=float)
    if hist.size == 0:
        # Create an empty plot for completeness
        plt.figure()
        plt.title("Energy trajectory (no samples recorded)")
        plt.xlabel("Sample index")
        plt.ylabel("Energy")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"[PLOT ] Saved empty history to {plot_path}")
        return
    plt.figure()
    plt.plot(np.arange(hist.size), hist)
    plt.xlabel("Sample index")
    plt.ylabel("Energy")
    plt.title("Energy trajectory (lower is better)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT ] {plot_path} (len={hist.size})")

def maybe_save_history(history: np.ndarray,
                       npy_path: Optional[str],
                       csv_path: Optional[str]) -> None:
    hist = np.asarray(history, dtype=float)
    if npy_path:
        np.save(npy_path, hist)
        print(f"[SAVE ] history NPY -> {npy_path} (len={hist.size})")
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "energy"])
            for i, e in enumerate(hist):
                w.writerow([i, float(e)])
        print(f"[SAVE ] history CSV -> {csv_path} (len={hist.size})")

def maybe_save_run_json(path: Optional[str],
                        mode: str,
                        args: argparse.Namespace,
                        qubo: Qubo,
                        cfg: VXQConfig,
                        wall_time_s: float,
                        final_energy: float,
                        dt: float,
                        artifacts: Dict[str, Optional[str]],
                        verify_info: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "mode": mode,  # "demo_maxcut" or "solve_qubo"
        "timestamp_utc_s": time.time(),
        "qubo": {
            "n": int(qubo.n),
            "has_b": qubo.b is not None,
        },
        "config": {
            "steps": int(cfg.steps),
            "restarts": int(cfg.restarts),
            "sample_every": int(cfg.sample_every),
            "beta0": float(cfg.beta0),
            "beta_max": float(cfg.beta_max),
            "noise0": float(cfg.noise0),
            "gamma": float(cfg.gamma),
            "dt_autotuned": cfg.dt is None,
            "plateau_patience": int(cfg.plateau_patience),
            "polish_iters": int(cfg.polish_iters),
            "seed": int(cfg.seed),
        },
        "args": vars(args),
        "results": {
            "final_energy": float(final_energy),
            "wall_time_s": float(wall_time_s),
            "integration_dt": float(dt),
            "local_opt_certificate_min_dE": float(verify_info.get("local_opt_min_dE", np.nan)),
            "exact": {
                "ran": bool(verify_info.get("exact_ran", False)),
                "opt_energy": (None if verify_info.get("exact_opt_energy") is None else float(verify_info["exact_opt_energy"])),
                "gap": (None if verify_info.get("exact_gap") is None else float(verify_info["exact_gap"])),
                "match_solution": (None if verify_info.get("exact_match") is None else bool(verify_info["exact_match"])),
                "time_s": (None if verify_info.get("exact_time_s") is None else float(verify_info["exact_time_s"])),
            }
        },
        "artifacts": artifacts,
        "versions": {
            "numpy": np.__version__,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[RUN  ] saved log -> {path}")


# ---------------------------
# CLI commands
# ---------------------------

def cmd_demo_maxcut(args: argparse.Namespace) -> None:
    W = random_maxcut_weights(args.n, args.p, seed=args.seed)
    qubo = maxcut_to_qubo(W)

    cfg = VXQConfig(
        steps=args.steps,
        restarts=args.restarts,
        sample_every=args.sample_every,
        beta0=args.beta0,
        beta_max=args.beta_max,
        noise0=args.noise0,
        gamma=args.gamma,
        dt=None if args.dt is None else float(args.dt),
        plateau_patience=args.plateau,
        polish_iters=args.polish,
        seed=args.seed,
        verbose=not args.quiet,
    )
    solver = VeloxQReconstructed(qubo, cfg)

    t0 = time.time()
    x, E, meta = solver.solve()
    t1 = time.time()

    cut_weight = -E
    print(f"\n[RESULT] n={qubo.n} | cut={cut_weight:.6f} | energy={E:.6f}")
    cert = local_optimality_certificate(qubo, x)
    print(f"[VERIFY] 1-opt certificate min ΔE = {cert:.6e} (>=0 ⇒ local optimum)")

    # Exact verification
    verify_info = {"local_opt_min_dE": cert}
    if args.exact and qubo.n <= args.nmax:
        start = time.time()
        exact = exact_min_qubo_gray(qubo.Q, qubo.b, nmax=args.nmax)
        took = time.time() - start
        if exact is not None:
            xb, Eb = exact
            cut_star = -Eb
            gap = cut_star - cut_weight
            print(f"[EXACT ] cut*={cut_star:.6f} | energy*={Eb:.6f} | gap={gap:+.6f} | time={took:.2f}s")
            print(f"[MATCH?] {np.array_equal(x, xb)}")
            verify_info.update({
                "exact_ran": True,
                "exact_opt_energy": Eb,
                "exact_gap": (-E) - (-Eb),  # same as Eb - E
                "exact_match": bool(np.array_equal(x, xb)),
                "exact_time_s": took,
            })
        else:
            print("[EXACT ] skipped (n too large for exact check).")
            verify_info.update({"exact_ran": False})
    else:
        verify_info.update({"exact_ran": False})

    # Plot / Save history if requested
    history = meta.get("history", np.array([]))
    maybe_plot_history(history, args.plot)
    maybe_save_history(history, args.save_history_npy, args.save_history_csv)

    # Artifacts dictionary
    artifacts = {
        "plot_path": args.plot,
        "history_npy": args.save_history_npy,
        "history_csv": args.save_history_csv,
        "solution_npy": None,  # not saved in demo_maxcut
        "qubo_npz": None,      # demo generates in-memory only
    }

    # Save run log JSON
    maybe_save_run_json(
        path=args.save_run_json,
        mode="demo_maxcut",
        args=args,
        qubo=qubo,
        cfg=cfg,
        wall_time_s=(t1 - t0),
        final_energy=E,
        dt=float(meta.get("dt", np.nan)),
        artifacts=artifacts,
        verify_info=verify_info,
    )


def cmd_solve_qubo(args: argparse.Namespace) -> None:
    qubo = load_qubo_npz(args.infile)
    cfg = VXQConfig(
        steps=args.steps,
        restarts=args.restarts,
        sample_every=args.sample_every,
        beta0=args.beta0,
        beta_max=args.beta_max,
        noise0=args.noise0,
        gamma=args.gamma,
        dt=None if args.dt is None else float(args.dt),
        plateau_patience=args.plateau,
        polish_iters=args.polish,
        seed=args.seed,
        verbose=not args.quiet,
    )
    solver = VeloxQReconstructed(qubo, cfg)

    t0 = time.time()
    x, E, meta = solver.solve()
    t1 = time.time()

    print(f"\n[RESULT] n={qubo.n} | energy={E:.6f}")
    cert = local_optimality_certificate(qubo, x)
    print(f"[VERIFY] 1-opt certificate min ΔE = {cert:.6e} (>=0 ⇒ local optimum)")

    # Save solution (optional)
    if args.out:
        np.save(args.out, x.astype(np.uint8))
        print(f"[SAVED ] {args.out}")

    # Exact verification
    verify_info = {"local_opt_min_dE": cert}
    if args.exact and qubo.n <= args.nmax:
        start = time.time()
        exact = exact_min_qubo_gray(qubo.Q, qubo.b, nmax=args.nmax)
        took = time.time() - start
        if exact is not None:
            xb, Eb = exact
            gap = E - Eb
            print(f"[EXACT ] E*={Eb:.6f} | gap={gap:+.6f} | time={took:.2f}s")
            print(f"[MATCH?] {np.allclose(E, Eb) and np.array_equal(x, xb)}")
            verify_info.update({
                "exact_ran": True,
                "exact_opt_energy": Eb,
                "exact_gap": E - Eb,
                "exact_match": bool(np.array_equal(x, xb)),
                "exact_time_s": took,
            })
        else:
            print("[EXACT ] skipped (n too large for exact check).")
            verify_info.update({"exact_ran": False})
    else:
        verify_info.update({"exact_ran": False})

    # Plot / Save history if requested
    history = meta.get("history", np.array([]))
    maybe_plot_history(history, args.plot)
    maybe_save_history(history, args.save_history_npy, args.save_history_csv)

    # Artifacts dictionary
    artifacts = {
        "plot_path": args.plot,
        "history_npy": args.save_history_npy,
        "history_csv": args.save_history_csv,
        "solution_npy": args.out,
        "qubo_npz": args.infile,
    }

    # Save run log JSON
    maybe_save_run_json(
        path=args.save_run_json,
        mode="solve_qubo",
        args=args,
        qubo=qubo,
        cfg=cfg,
        wall_time_s=(t1 - t0),
        final_energy=E,
        dt=float(meta.get("dt", np.nan)),
        artifacts=artifacts,
        verify_info=verify_info,
    )


def cmd_verify(args: argparse.Namespace) -> None:
    qubo = load_qubo_npz(args.infile)
    x = np.load(args.xfile).astype(int).reshape(qubo.n)
    E = qubo.energy(x)
    cert = local_optimality_certificate(qubo, x)
    print(f"[VERIFY] energy={E:.6f} | 1-opt min ΔE={cert:.6e} (>=0 ⇒ local optimum)")
    if args.exact and qubo.n <= args.nmax:
        start = time.time()
        exact = exact_min_qubo_gray(qubo.Q, qubo.b, nmax=args.nmax)
        took = time.time() - start
        if exact is not None:
            xb, Eb = exact
            same = np.array_equal(x, xb)
            print(f"[EXACT ] E*={Eb:.6f} | same_solution={same} | ΔE={E - Eb:+.6f} | time={took:.2f}s")
        else:
            print("[EXACT ] skipped (n too large for exact check).")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VeloxQ-style (reconstructed) QUBO solver with verification, plotting, history export, and run logging.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # demo_maxcut
    pd = sub.add_parser("demo_maxcut", help="Generate a random MAX-CUT instance and solve.")
    pd.add_argument("--n", type=int, default=40, help="Number of nodes.")
    pd.add_argument("--p", type=float, default=0.25, help="Edge probability (Erdos–Renyi).")
    pd.add_argument("--seed", type=int, default=42)
    pd.add_argument("--steps", type=int, default=5000)
    pd.add_argument("--restarts", type=int, default=3)
    pd.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    pd.add_argument("--beta0", type=float, default=0.1)
    pd.add_argument("--beta-max", dest="beta_max", type=float, default=8.0)
    pd.add_argument("--noise0", type=float, default=0.01)
    pd.add_argument("--gamma", type=float, default=0.2)
    pd.add_argument("--dt", type=float, default=None, help="Override auto-tuned dt (advanced).")
    pd.add_argument("--plateau", type=int, default=200, help="Steps with no improvement before a kick.")
    pd.add_argument("--polish", type=int, default=2000, help="Greedy 1-opt iterations.")
    pd.add_argument("--exact", action="store_true", help="Run exact verification if n<=nmax.")
    pd.add_argument("--nmax", type=int, default=22, help="Max n for exact check.")
    pd.add_argument("--plot", type=str, default=None, help="Save energy history plot to this path (e.g., energy.png).")
    pd.add_argument("--save-history-npy", dest="save_history_npy", type=str, default=None, help="Save energy history to .npy")
    pd.add_argument("--save-history-csv", dest="save_history_csv", type=str, default=None, help="Save energy history to .csv")
    pd.add_argument("--save-run-json", dest="save_run_json", type=str, default=None, help="Save a JSON log of this run (config, results, artifacts)")
    pd.add_argument("--quiet", action="store_true")

    # solve_qubo
    ps = sub.add_parser("solve_qubo", help="Solve a saved QUBO (npz with Q, optional b).")
    ps.add_argument("--in", dest="infile", required=True)
    ps.add_argument("--steps", type=int, default=5000)
    ps.add_argument("--restarts", type=int, default=3)
    ps.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    ps.add_argument("--beta0", type=float, default=0.1)
    ps.add_argument("--beta-max", dest="beta_max", type=float, default=8.0)
    ps.add_argument("--noise0", type=float, default=0.01)
    ps.add_argument("--gamma", type=float, default=0.2)
    ps.add_argument("--dt", type=float, default=None)
    ps.add_argument("--plateau", type=int, default=200)
    ps.add_argument("--polish", type=int, default=2000)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--out", type=str, default=None, help="Save binary solution to .npy")
    ps.add_argument("--exact", action="store_true")
    ps.add_argument("--nmax", type=int, default=22)
    ps.add_argument("--plot", type=str, default=None, help="Save energy history plot to this path (e.g., energy.png).")
    ps.add_argument("--save-history-npy", dest="save_history_npy", type=str, default=None, help="Save energy history to .npy")
    ps.add_argument("--save-history-csv", dest="save_history_csv", type=str, default=None, help="Save energy history to .csv")
    ps.add_argument("--save-run-json", dest="save_run_json", type=str, default=None, help="Save a JSON log of this run (config, results, artifacts)")
    ps.add_argument("--quiet", action="store_true")

    # verify
    pv = sub.add_parser("verify", help="Verify/evaluate a solution against a QUBO.")
    pv.add_argument("--in", dest="infile", required=True)
    pv.add_argument("--x", dest="xfile", required=True)
    pv.add_argument("--exact", action="store_true")
    pv.add_argument("--nmax", type=int, default=22)

    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.cmd == "demo_maxcut":
        cmd_demo_maxcut(args)
    elif args.cmd == "solve_qubo":
        cmd_solve_qubo(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    else:
        parser.error(f"unknown cmd {args.cmd!r}")


if __name__ == "__main__":
    main()
