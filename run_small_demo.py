#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot demo:
- writes example_qubo.npz (6-variable MAX-CUT QUBO)
- runs the reconstructed VeloxQ solver
- saves sol.npy
- verifies solution (exact enumeration + 1-opt certificate)
"""

import numpy as np
import time

# Import the solver & helpers from the file you saved earlier
from veloxq_reconstructed import (
    Qubo,
    VXQConfig,
    VeloxQReconstructed,
    save_qubo_npz,
    load_qubo_npz,
    exact_min_qubo_gray,
    local_optimality_certificate,
)

def write_example_qubo(path: str = "example_qubo.npz") -> None:
    # Exact arrays (6-variable MAX-CUT-derived QUBO)
    Q = np.array([
        [0.     , 0.18727, 0.73239, 0.47088, 0.47445, 0.48639],
        [0.18727, 0.     , 0.42527, 0.81272, 0.20737, 0.34842],
        [0.73239, 0.42527, 0.     , 0.53986, 0.70310, 0.46365],
        [0.47088, 0.81272, 0.53986, 0.     , 0.91825, 0.52088],
        [0.47445, 0.20737, 0.70310, 0.91825, 0.     , 0.55276],
        [0.48639, 0.34842, 0.46365, 0.52088, 0.55276, 0.     ],
    ], dtype=float)

    # b = -row sums (MAX-CUT → QUBO derivation)
    b = np.array([
        -Q[0].sum(),
        -Q[1].sum(),
        -Q[2].sum(),
        -Q[3].sum(),
        -Q[4].sum(),
        -Q[5].sum(),
    ], dtype=float)

    save_qubo_npz(path, Q, b)
    print(f"[WRITE] Saved {path} with n=6")

def solve_and_verify(infile: str = "example_qubo.npz", out_sol: str = "sol.npy") -> None:
    # Load QUBO
    qubo: Qubo = load_qubo_npz(infile)
    n = qubo.n
    print(f"[LOAD ] {infile} | n={n}")

    # Configure solver (tweak steps/restarts if you want)
    cfg = VXQConfig(
        steps=4000,
        restarts=3,
        sample_every=10,
        beta0=0.1,
        beta_max=8.0,
        noise0=0.01,
        gamma=0.2,
        dt=None,            # auto-tune from spectral radius
        plateau_patience=200,
        polish_iters=2000,
        seed=123,
        verbose=True,
    )

    solver = VeloxQReconstructed(qubo, cfg)

    # Solve
    t0 = time.time()
    x_bin, energy, meta = solver.solve()
    t1 = time.time()
    print(f"[RESULT] energy={energy:.6f} | time={t1 - t0:.3f}s")

    # Save solution
    np.save(out_sol, x_bin.astype(np.uint8))
    print(f"[SAVE ] {out_sol} (binary vector shape={x_bin.shape})")

    # Local optimality certificate (1-opt)
    cert = local_optimality_certificate(qubo, x_bin)
    print(f"[CHECK ] 1-opt min ΔE = {cert:.6e} (>=0 ⇒ local optimum)")

    # Exact verification (n=6 → feasible)
    start = time.time()
    exact = exact_min_qubo_gray(qubo.Q, qubo.b, nmax=22)
    took = time.time() - start
    if exact is None:
        print("[EXACT ] Skipped (n too large).")
    else:
        x_star, E_star = exact
        match = np.array_equal(x_bin, x_star)
        gap = energy - E_star
        print(f"[EXACT ] E*={E_star:.6f} | gap={gap:+.6f} | match={match} | time={took:.3f}s")

def main():
    write_example_qubo("example_qubo.npz")
    solve_and_verify("example_qubo.npz", "sol.npy")

if __name__ == "__main__":
    main()
