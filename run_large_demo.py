#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-scale demo (n=200) for the reconstructed VeloxQ-style QUBO solver.

What it does:
- Builds a random MAX-CUT instance (n=200) with Erdos–Renyi edges and random weights.
- Converts to QUBO, runs the VeloxQ-style solver with multiple restarts.
- Saves:
    * Binary solution:            sol_200.npy
    * Energy trajectory (np):     energy_history.npy
    * Energy trajectory (csv):    energy_history.csv
    * Plot of energy over time:   energy_history.png
- Prints local 1-opt certificate and final energy.

Requirements:
    pip install numpy matplotlib
"""

import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from veloxq_reconstructed import (
    Qubo,
    VXQConfig,
    VeloxQReconstructed,
    maxcut_to_qubo,
    random_maxcut_weights,
    save_qubo_npz,
    load_qubo_npz,
    local_optimality_certificate,
)

def build_maxcut_qubo(n=200, p=0.10, seed=123, out_npz="large_qubo_200.npz") -> Qubo:
    """Generate and persist a reproducible MAX-CUT instance as a QUBO."""
    W = random_maxcut_weights(n, p, seed=seed)
    qubo = maxcut_to_qubo(W)
    save_qubo_npz(out_npz, qubo.Q, qubo.b)
    print(f"[WRITE] Saved {out_npz} (n={n}, p={p}, seed={seed})")
    return qubo

def solve_large(qubo: Qubo,
                steps=8000,
                restarts=5,
                seed=123,
                polish_iters=3000,
                out_sol="sol_200.npy",
                out_hist_npy="energy_history.npy",
                out_hist_csv="energy_history.csv",
                out_plot="energy_history.png"):
    """Run solver, save artifacts, and plot energy trajectory."""
    cfg = VXQConfig(
        steps=steps,
        restarts=restarts,
        sample_every=10,     # record energy every 10 steps
        beta0=0.1,
        beta_max=8.0,
        noise0=0.01,
        gamma=0.2,
        dt=None,            # auto-tune by spectral radius
        plateau_patience=250,
        polish_iters=polish_iters,
        seed=seed,
        verbose=True,
    )

    solver = VeloxQReconstructed(qubo, cfg)

    t0 = time.time()
    x_bin, energy, meta = solver.solve()
    t1 = time.time()
    print(f"[RESULT] n={qubo.n} | energy={energy:.6f} | elapsed={t1 - t0:.2f}s")

    # Save solution vector
    np.save(out_sol, x_bin.astype(np.uint8))
    print(f"[SAVE ] {out_sol} (shape={x_bin.shape})")

    # Save energy history (np + csv)
    hist = np.asarray(meta.get("history", []), dtype=float)
    np.save(out_hist_npy, hist)
    print(f"[SAVE ] {out_hist_npy} (len={hist.size})")

    with open(out_hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "energy"])
        for i, e in enumerate(hist):
            w.writerow([i, float(e)])
    print(f"[SAVE ] {out_hist_csv}")

    # Local 1-opt certificate
    cert = local_optimality_certificate(qubo, x_bin)
    print(f"[CHECK] 1-opt min ΔE = {cert:.6e} (>=0 ⇒ local optimum)")

    # Plot energy trajectory
    if hist.size > 0:
        plt.figure()
        plt.plot(np.arange(hist.size), hist)
        plt.xlabel("Sample index")
        plt.ylabel("Energy")
        plt.title("Energy trajectory (lower is better)")
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        print(f"[PLOT] {out_plot}")

def main():
    # 1) Build a large random instance (adjust p for sparsity/density)
    qubo = build_maxcut_qubo(n=200, p=0.10, seed=123, out_npz="large_qubo_200.npz")

    # 2) Solve and produce artifacts
    solve_large(
        qubo=qubo,
        steps=8000,        # increase for higher quality
        restarts=5,        # more restarts -> better chance for low energy
        seed=123,
        polish_iters=3000, # aggressive 1-opt polish
        out_sol="sol_200.npy",
        out_hist_npy="energy_history.npy",
        out_hist_csv="energy_history.csv",
        out_plot="energy_history.png",
    )

if __name__ == "__main__":
    main()
