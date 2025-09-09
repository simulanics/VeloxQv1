#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-accuracy VeloxQ-style feature selection on 20 Newsgroups
-------------------------------------------------------------
- TF-IDF (1-2 grams), sublinear TF, df filtering, numeric/junky tokens removed
- χ² scores computed on binary presence
- χ² pre-pruning (keep top M features) before QUBO
- Sparse redundancy penalty (cosine similarity, top-K neighbors, safe-normalized)
- FULL k-constraint with dense 11^T + adaptive P (weaker start so it doesn't dominate)
- Global energy scaling + dt override for stability on large dense Q
- Sweep across several k values; evaluate LogisticRegression and LinearSVC (small C-grids)
- Pick the best validation accuracy; export PNG + detailed HTML report
"""

import base64, io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from veloxq_reconstructed import (
    Qubo, VXQConfig, VeloxQReconstructed, local_optimality_certificate
)

# ---------------------------
# Global config (tune here)
# ---------------------------
CATEGORIES      = ["sci.space", "comp.graphics", "rec.sport.hockey"]

# Vectorizer: tighter vocab (remove numeric/junk, stricter df)
NGRAMS          = (1, 2)
MAX_FEATURES    = 3000
MIN_DF          = 5
MAX_DF          = 0.85
TOKEN_PATTERN   = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # drop digits & single letters

TEST_SIZE       = 0.30
RANDOM_STATE    = 42

# χ² pre-pruning before QUBO
PRE_KEEP        = 2500  # keep top-M features by χ² before building redundancy/QUBO

# QUBO weights (stronger reward, slightly higher redundancy, weaker k-constraint)
ALPHA           = 6.0     # reward weight for χ² score
BETA            = 0.005   # mild per-feature penalty
LAMBDA_RED      = 0.30    # redundancy penalty multiplier
TOPK_RED        = 30      # top-K cosine neighbors per feature
ADAPT_TOL       = 0.15    # ±15% band for adaptive P hits
ADAPT_MAX_TRIES = 6
P_START         = 0.02    # weaker k-constraint to avoid dominating energy

# Solver config (large dense Q needs conservative step sizes)
STEPS           = 7000
RESTARTS        = 6
POLISH_ITERS    = 4000
SEED            = 123
BETA_MAX_SOLV   = 12.0
PLATEAU         = 150
DT_OVERRIDE     = 0.0015  # stability for large dense Q

# Energy scaling for stability (keeps argmin unchanged)
ENERGY_SCALE    = 5e-4

# k sweep and classifiers
K_GRID          = [50, 100, 150, 200, 300, 400, 500]

# Output
PLOT_PATH       = "selected_features.png"
REPORT_PATH     = "feature_selection_report.html"


# ---------------------------
# Utilities
# ---------------------------
def normalize_01(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (np.ptp(arr) + 1e-12)

def build_redundancy_matrix_safe(X_train: np.ndarray, topk: int) -> np.ndarray:
    """
    Cosine similarity between feature columns; safe-normalize columns
    (no divide-by-zero), keep only top-K per column; symmetrize and sanitize.
    """
    norms = np.linalg.norm(X_train, axis=0)
    norms_safe = np.where(norms > 0.0, norms, 1.0)
    Xn = X_train / norms_safe  # no NaNs/inf

    S = (Xn.T @ Xn).astype(float)
    np.fill_diagonal(S, 0.0)

    d = S.shape[0]
    if topk < d:
        for j in range(d):
            cut = np.argpartition(S[:, j], -topk)[:-topk]
            S[cut, j] = 0.0

    S = np.maximum(S, S.T)
    np.clip(S, -1.0, 1.0, out=S)
    S[~np.isfinite(S)] = 0.0
    return S

def qubo_base(scores01: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Base Q without k-constraint: diag(-ALPHA*scores + BETA) + LAMBDA_RED * S
    """
    diag = -ALPHA * scores01 + BETA
    Q = np.diag(diag)
    Q += LAMBDA_RED * S
    Q[~np.isfinite(Q)] = 0.0
    return Q

def add_k_constraint(Q_base: np.ndarray, k: int, P: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Q, b) adding FULL constraint P*(sum x - k)^2:
      Q += P * 11^T
      b  = -2*k*P * 1
    Then scale by ENERGY_SCALE and sanitize to finite.
    """
    d = Q_base.shape[0]
    Q = Q_base + P * np.ones((d, d), dtype=float)
    b = np.full(d, -2.0 * k * P, dtype=float)

    Q[~np.isfinite(Q)] = 0.0
    b[~np.isfinite(b)] = 0.0

    if ENERGY_SCALE != 1.0:
        Q *= ENERGY_SCALE
        b *= ENERGY_SCALE
    return Q, b

def solve_for_k(Q_base: np.ndarray, k: int, P_start: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Adaptive P loop to hit ~k features. Returns (x_bin, sel_idx, history, energy, cert, P_used).
    Uses a conservative dt override + energy scaling for stability on large dense QUBOs.
    """
    P = float(P_start)
    best_pack = None  # (gap_to_k, x_bin, sel_idx, energy, cert, P)
    history = np.array([], float)

    for _ in range(ADAPT_MAX_TRIES):
        Q, b = add_k_constraint(Q_base, k, P)
        if not (np.isfinite(Q).all() and np.isfinite(b).all()):
            raise ValueError("Non-finite entries in Q/b after construction.")

        qubo = Qubo.from_arrays(Q, b)
        cfg = VXQConfig(steps=STEPS, restarts=RESTARTS, polish_iters=POLISH_ITERS,
                        seed=SEED, verbose=True, beta_max=BETA_MAX_SOLV,
                        plateau_patience=PLATEAU, dt=DT_OVERRIDE)
        solver = VeloxQReconstructed(qubo, cfg)
        x_bin, energy, meta = solver.solve()
        hist = meta.get("history", np.array([], float))
        history = np.concatenate([history, hist]) if hist.size else history

        sel = np.flatnonzero(x_bin)
        m = len(sel)
        cert = local_optimality_certificate(qubo, x_bin)

        gap = abs(m - k)
        if (best_pack is None) or (gap < best_pack[0]):
            best_pack = (gap, x_bin, sel, float(energy), float(cert), P)

        if (1 - ADAPT_TOL) * k <= m <= (1 + ADAPT_TOL) * k:
            break
        P = P * 1.8 if m > k else P / 1.8

    _, x_best, sel_best, E_best, cert_best, P_used = best_pack
    return x_best, sel_best, history, E_best, cert_best, P_used

def eval_models(X_tr, y_tr, X_va, y_va) -> Tuple[str, float]:
    """
    Evaluate LogisticRegression and LinearSVC over small C-grids; return (best_model_name, best_acc).
    """
    best_name, best_acc = "baseline", -1.0

    # Logistic Regression grid
    for C in (1.0, 2.0, 3.0, 5.0):
        lr = LogisticRegression(max_iter=4000, solver="lbfgs", C=C)
        lr.fit(X_tr, y_tr)
        acc = accuracy_score(y_va, lr.predict(X_va))
        if acc > best_acc:
            best_acc = acc
            best_name = f"LogisticRegression(C={C})"

    # Linear SVM grid
    for C in (0.5, 1.0, 2.0, 3.0):
        svm = LinearSVC(C=C)
        svm.fit(X_tr, y_tr)
        acc = accuracy_score(y_va, svm.predict(X_va))
        if acc > best_acc:
            best_acc = acc
            best_name = f"LinearSVC(C={C})"

    return best_name, float(best_acc)

def make_plot(scores01: np.ndarray, selected_idx: np.ndarray, title: str, out_path: str) -> str:
    plt.figure(figsize=(12, 5))
    x = np.arange(scores01.size)
    plt.bar(x, scores01, alpha=0.6, label="All features")
    if selected_idx.size > 0:
        plt.bar(selected_idx, scores01[selected_idx], alpha=0.9, label="Selected features")
    plt.xlabel("Feature index")
    plt.ylabel("χ² score (normalized)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.savefig(out_path, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def html_escape(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            .replace('"',"&quot;").replace("'","&#39;"))


# ---------------------------
# 1) Load & vectorize
# ---------------------------
print("[DATA] Loading 20 Newsgroups…")
data = fetch_20newsgroups(
    subset="train", categories=CATEGORIES, remove=("headers","footers","quotes")
)
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=NGRAMS,
    min_df=MIN_DF,
    max_df=MAX_DF,
    sublinear_tf=True,
    norm="l2",
    token_pattern=TOKEN_PATTERN,   # drop numeric & single-letter tokens
)
X = vectorizer.fit_transform(data.data).toarray()
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
d_full = X.shape[1]
feature_names_full = np.array(vectorizer.get_feature_names_out())

# χ² on binary presence (use training split)
X_train_bin = (X_train > 0).astype(np.int32)
chi2_scores, _ = chi2(X_train_bin, y_train)
scores01_all = normalize_01(chi2_scores)

# ---------------------------
# 1.5) χ² pre-pruning before QUBO
# ---------------------------
order_all = np.argsort(-scores01_all)
keep_idx = order_all[:min(PRE_KEEP, d_full)]

X_train = X_train[:, keep_idx]
X_val   = X_val[:, keep_idx]
scores01 = scores01_all[keep_idx]
feature_names = feature_names_full[keep_idx]
d = scores01.size

# ---------------------------
# Redundancy matrix (safe) and base Q
# ---------------------------
print("[RED ] Building redundancy matrix (cosine, top-K)…")
S = build_redundancy_matrix_safe(X_train, topk=TOPK_RED)

Q_base = qubo_base(scores01, S)

# ---------------------------
# 2) Sweep over k, keep best by validation accuracy
# ---------------------------
results = []  # (k, selected_count, best_model, acc, energy, cert, P_used)
best_acc = -1.0
best_pack = None  # (k, sel_idx, energy, cert, P_used)

for k in K_GRID:
    print(f"\n[SWEEP] k={k}")
    x_bin, sel_idx, hist, E, cert, P_used = solve_for_k(Q_base, k, P_START)
    m = int(sel_idx.size)
    sel_idx = sel_idx if m > 0 else np.array([0], dtype=int)  # avoid degenerate slice

    model_name, acc = eval_models(X_train[:, sel_idx], y_train, X_val[:, sel_idx], y_val)
    results.append((k, m, model_name, acc, E, cert, P_used))
    if acc > best_acc:
        best_acc = acc
        best_pack = (k, sel_idx, E, cert, P_used)

# Use best selection for visualization & final output
K_best, selected, energy_best, cert_best, P_used_best = best_pack
selected_count = int(selected.size)
best_title = f"Selected {selected_count} features (k*={K_best}) out of {d}"
img_b64 = make_plot(scores01, selected, best_title, PLOT_PATH)

# Evaluate models again on best set for the report
model_name_best, acc_best = eval_models(X_train[:, selected], y_train, X_val[:, selected], y_val)

# Rank top selected tokens by χ² score
order = selected[np.argsort(-scores01[selected])] if selected_count else np.array([], int)
top_words = feature_names[order][:50]

print(f"\n[RESULT] k*={K_best} | selected={selected_count}/{d} | best_model={model_name_best} | acc={acc_best:.4f}")
print(f"         energy={energy_best:.6f} | 1-opt cert={cert_best:.6g} | P_used={P_used_best}")
print("\nTop selected words:\n", top_words)

# ---------------------------
# 3) HTML report
# ---------------------------
def table(rows: List[Tuple[str, str]]) -> str:
    return "<table>" + "".join(
        f"<tr><th align='left'>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>"
        for k, v in rows
    ) + "</table>"

params_rows = [
    ("Categories", ", ".join(CATEGORIES)),
    ("Vectorizer", f"TF-IDF, ngrams={NGRAMS}, token_pattern='{TOKEN_PATTERN}', sublinear_tf=True, min_df={MIN_DF}, max_df={MAX_DF}"),
    ("Max features (pre-vectorizer)", str(MAX_FEATURES)),
    ("χ² pre-pruning kept", f"{d} / {d_full}"),
    ("Train/Val split", f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}%"),
    ("ALPHA (score reward)", f"{ALPHA}"),
    ("BETA (per-feature)", f"{BETA}"),
    ("LAMBDA_RED (redundancy)", f"{LAMBDA_RED}"),
    ("TOPK_RED", f"{TOPK_RED}"),
    ("Energy scale", f"{ENERGY_SCALE}"),
    ("Solver steps/restarts/polish", f"{STEPS}/{RESTARTS}/{POLISH_ITERS}"),
    ("beta_max / plateau / dt", f"{BETA_MAX_SOLV} / {PLATEAU} / {DT_OVERRIDE}"),
    ("Seed", f"{SEED}"),
]

results_header = "<tr><th>k</th><th>|sel|</th><th>Best model</th><th>Acc</th><th>Energy</th><th>1-opt ΔE</th><th>P used</th></tr>"
results_rows = "".join(
    f"<tr><td>{k}</td><td>{m}</td><td>{html_escape(model)}</td><td>{acc:.4f}</td>"
    f"<td>{E:.3f}</td><td>{cert:.6g}</td><td>{P_used:.4g}</td></tr>"
    for (k, m, model, acc, E, cert, P_used) in results
)
results_table = f"<table>{results_header}{results_rows}</table>"

top_words_html = "<br>".join(html_escape(w) for w in top_words)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>VeloxQ-style Feature Selection Report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: .2em; }} small {{ color:#666; }}
table {{ border-collapse: collapse; margin: 10px 0 20px; }}
th,td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
.figure {{ margin-top: 18px; }}
hr {{ border: none; border-top: 1px solid #eee; margin: 24px 0; }}
code {{ background:#f7f7f7; padding:2px 4px; border-radius:4px; }}
</style></head><body>

<h1>Feature Selection Report</h1>
<small>VeloxQ-style QUBO solver (reconstruction)</small>

<h2>Parameters</h2>
{table(params_rows)}

<h2>k-sweep results</h2>
{results_table}

<h2>Best selection</h2>
{table([
    ("k*", str(K_best)),
    ("Selected features", f"{selected_count} / {d}"),
    ("Best model", model_name_best),
    ("Validation accuracy", f"{acc_best:.6f}"),
    ("Final energy", f"{energy_best:.6f}"),
    ("Local-opt certificate (min ΔE)", f"{cert_best:.6g}"),
    ("P used", f"{P_used_best:.6g}"),
])}

<h3>Top selected tokens (by χ²)</h3>
<p>{top_words_html}</p>

<div class="figure">
<h3>Selection overlay</h3>
<img alt="Selected features overlay" src="data:image/png;base64,{img_b64}" />
</div>

<hr>
<p><strong>Notes.</strong> We pre-prune by χ², build QUBO
diag(<code>-α·χ²_norm + β</code>) + <code>λ_red·S</code> (sparse cosine-sim redundancy),
add a full k-constraint <code>P·11ᵀ</code> with linear term <code>-2kP·1</code>, scale energies
by <code>{ENERGY_SCALE}</code>, and use a conservative timestep <code>dt={DT_OVERRIDE}</code>.
We adapt P to keep |selected| ≈ k, sweep several k, and keep the classifier (LR or LinearSVC)
with the best validation accuracy.</p>

</body></html>
"""

Path(REPORT_PATH).write_text(html, encoding="utf-8")
print(f"\n[FILES] Plot: {PLOT_PATH}\n[FILES] Report: {REPORT_PATH}\n")
