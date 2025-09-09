#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection with VeloxQ-Style QUBO (Cardinality + Diversity)
------------------------------------------------------------------
We construct a QUBO that:
  (1) Rewards features that reduce validation log-loss vs. a baseline prior.
  (2) Softly enforces ~k selected features using λ * (Σ x - k)^2.
  (3) Penalizes redundant pairs using an |corr| term.

Energy (to MINIMIZE):
  E(x) = sum_i [ (-alpha * gain_i) + beta ] * x_i
       + lambda_card * (Σ_i x_i - k)^2
       + eta_redundancy * Σ_{i<j} |corr_ij| * x_i x_j

Mapping into QUBO (x^T Q x + b^T x):
  - Linear terms go on the diagonal (since x_i^2 = x_i for binary).
  - (Σ x - k)^2 expands to: (1-2k)Σ x_i + 2Σ_{i<j} x_i x_j (+ const).
    -> Add lambda_card*(1-2k) to Q_ii and lambda_card to Q_ij (i≠j).
  - Redundancy: add 0.5 * eta_redundancy * |corr_ij| to Q_ij (i≠j)
    because x^T Q x contributes 2*Q_ij*x_i*x_j for i<j.

This yields a non-trivial optimum near k selected features.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from veloxq_reconstructed import (
    Qubo,
    VXQConfig,
    VeloxQReconstructed,
    save_qubo_npz,
    load_qubo_npz,
    local_optimality_certificate
)

# ---------------------------
# 1) Synthetic data
# ---------------------------
X, y = make_classification(
    n_samples=500,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, random_state=42
)

n_features = X.shape[1]

# ---------------------------
# 2) QUBO hyperparameters
# ---------------------------
# Reward strength for informative features (via loss gain)
alpha = 1.0

# Per-feature L1-like sparsity pressure (small; main sparsity is via cardinality)
beta = 0.00

# Soft exact-k penalty strength
lambda_card = 0.50     # try 0.25–1.0; higher = sharper pull toward k

# Target number of features to select
target_k = 4           # with 6 informative features, ~3–5 is sensible

# Redundancy (pairwise correlation) penalty
eta_redundancy = 0.10  # try 0.05–0.25; penalizes selecting collinear pairs

# ---------------------------
# 3) Feature gains vs baseline
# ---------------------------
# Baseline model: predict class prior probability (no features)
p_base = float(np.mean(y_train))
P_base = np.column_stack([
    np.repeat(1.0 - p_base, len(y_val)),
    np.repeat(p_base,       len(y_val))
])
baseline_loss = log_loss(y_val, P_base)

# Single-feature validation losses + gains
feature_losses = []
for j in range(n_features):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train[:, [j]], y_train)
    y_proba = clf.predict_proba(X_val[:, [j]])
    loss_j = log_loss(y_val, y_proba)
    feature_losses.append(loss_j)
feature_losses = np.array(feature_losses, dtype=float)

# Gain: how much better than the baseline (higher is better)
# (Can be negative if a feature is worse than baseline.)
gains = baseline_loss - feature_losses

# ---------------------------
# 4) Redundancy matrix (|corr|)
# ---------------------------
# Use training set correlations (more stable)
corr = np.corrcoef(X_train, rowvar=False)  # (n_features, n_features)
corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
corr_abs = np.abs(corr)
np.fill_diagonal(corr_abs, 0.0)  # no self-penalty on the diagonal

# ---------------------------
# 5) Build QUBO: Q (n x n), b=None
# ---------------------------
Q = np.zeros((n_features, n_features), dtype=float)

# Diagonal: per-feature rewards + base sparsity + cardinality linear part
#   diag_i = (-alpha * gain_i) + beta + lambda_card * (1 - 2k)
diag = (-alpha * gains) + beta + lambda_card * (1.0 - 2.0 * target_k)
np.fill_diagonal(Q, diag)

# Off-diagonal: exact-k pairwise term + redundancy penalty
#   For i!=j: Q_ij += lambda_card   (→ 2*lambda_card*x_i*x_j)
#             Q_ij += 0.5 * eta_redundancy * |corr_ij|
#   (the 0.5 factor gives total coefficient eta*|corr_ij| in x^T Q x)
# Start with the cardinality part:
Q += lambda_card * (np.ones((n_features, n_features)) - np.eye(n_features))

# Add redundancy (ensure zero diagonal)
Q += 0.5 * eta_redundancy * corr_abs

# No explicit linear 'b' (we encoded linear parts in Q_ii)
b = None

# Package & save
qubo = Qubo.from_arrays(Q, b)
save_qubo_npz("feature_select_qubo.npz", Q, b)

# ---------------------------
# 6) Solve with VeloxQ
# ---------------------------
cfg = VXQConfig(
    steps=4000,
    restarts=5,
    polish_iters=2000,
    seed=123,
    verbose=True
)
solver = VeloxQReconstructed(qubo, cfg)

x_bin, energy, meta = solver.solve()
selected = np.where(x_bin == 1)[0]

print("\nSelected features:", selected.tolist())
print("Selected count:", len(selected), f"(target_k={target_k})")
print("Final QUBO energy:", energy)
print("1-opt certificate:", local_optimality_certificate(qubo, x_bin))

# ---------------------------
# 7) Evaluate downstream model
# ---------------------------
if len(selected) == 0:
    raise RuntimeError(
        "No features selected. Increase lambda_card or lower beta / adjust alpha."
    )

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train[:, selected], y_train)
y_pred = clf.predict(X_val[:, selected])
acc = accuracy_score(y_val, y_pred)
print("Validation accuracy with selected features:", acc)

# Optional: show per-feature stats
print("\nPer-feature diagnostics (j, gain, loss):")
for j in range(n_features):
    mark = "*" if j in selected else " "
    print(f" {mark} j={j:2d} | gain={gains[j]:+.4f} | loss={feature_losses[j]:.4f}")
