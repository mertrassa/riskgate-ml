"""
Optimal depth-3 decision tree by Mixed-Integer Optimization (MIO), in the
spirit of Bertsimas & Dunn, "Optimal Classification Trees" (2017), but with
the objective and constraints the chip actually cares about:

  maximise   anomalous orders blocked by the ML that the hard rules let through
  subject to legit orders blocked only by the ML  <=  budget x (all legit orders)
             a full depth-3 tree, univariate integer splits  x_f > t  (as in silicon)

CART grows the tree greedily, one split at a time, and optimises impurity.
MIO picks all 7 splits and 8 leaf labels jointly, for the real objective.

Formulation (only z and c are integer):
  z[n,s] in {0,1}  node n uses split s = (feature, threshold)   sum_s z[n,s] = 1
  c[l]   in {0,1}  leaf l is labelled anomalous
  w[p,l] in [0,1]  cell p reaches leaf l; sum_l w[p,l] = 1 and, for each
                   ancestor (n, dir) of l,  w[p,l] <= sum_{s routes p to dir} z[n,s]
                   -> for integral z exactly one leaf is feasible, so w is exact.
  a[p,l] <= w[p,l], a[p,l] <= c[l]           (anomalies caught)
  f[p,l] >= w[p,l] + c[l] - 1                 (legit orders blocked)

Points with identical routing under every candidate split are merged into
weighted cells, which keeps the model small.  Solved with HiGHS via SciPy.
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from riskgate_model import Tree

# ancestors of leaf l in heap order: [(node, dir), ...] for depths 0,1,2
ANCESTORS = []
for leaf in range(8):
    b = [(leaf >> 2) & 1, (leaf >> 1) & 1, leaf & 1]
    n, path = 0, []
    for d in b:
        path.append((n, d))
        n = 2 * n + 1 + d
    ANCESTORS.append(path)


def candidate_splits(X, k, extra=()):
    """Per-feature integer thresholds from quantiles, plus any extra (f, t)."""
    splits = set(extra)
    for f in range(X.shape[1]):
        vals = X[:, f]
        qs = np.unique(np.floor(np.quantile(vals, np.linspace(0, 1, k + 2)[1:-1])).astype(int))
        splits.update((f, int(t)) for t in qs if 0 <= t < 255)
    return sorted(splits)


def fit_mio(X, y, n_legit_total, budget, k=12, extra=(), time_limit=300, verbose=False):
    """X, y: orders the hard rules ACCEPTED (features as the chip sees them).
    n_legit_total: number of all legit orders (rule-rejected ones included)."""
    splits = candidate_splits(X, k, extra)
    S = len(splits)
    # routing matrix R[i, s] = 1 if order i goes right on split s
    R = np.stack([X[:, f] > t for f, t in splits], axis=1)
    # merge orders with identical routing
    keys, inv = np.unique(R, axis=0, return_inverse=True)
    inv = inv.ravel()
    P = len(keys)
    n_anom = np.bincount(inv, weights=(y == 1), minlength=P)
    n_legit = np.bincount(inv, weights=(y == 0), minlength=P)
    always_left = S  # extra split index: (feature 0, threshold 255)
    S1 = S + 1
    keys = np.hstack([keys, np.zeros((P, 1), dtype=bool)])

    # variable layout
    iz = lambda n, s: n * S1 + s
    ic = lambda l: 7 * S1 + l
    base_w = 7 * S1 + 8
    iw = lambda p, l: base_w + p * 8 + l
    base_a = base_w + P * 8
    iA = lambda p, l: base_a + p * 8 + l
    base_f = base_a + P * 8
    iF = lambda p, l: base_f + p * 8 + l
    nvar = base_f + P * 8

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, lb, ub):
        nonlocal r
        for c_, v in entries:
            rows.append(r); cols.append(c_); vals.append(v)
        lo.append(lb); hi.append(ub)
        r += 1

    for n in range(7):                                   # one split per node
        add([(iz(n, s), 1) for s in range(S1)], 1, 1)
    for p in range(P):
        add([(iw(p, l), 1) for l in range(8)], 1, 1)     # each cell in one leaf
        right = np.flatnonzero(keys[p])
        left = np.flatnonzero(~keys[p])
        for l in range(8):
            for n, d in ANCESTORS[l]:                    # w <= sum of compatible splits
                compat = right if d else left
                add([(iw(p, l), 1)] + [(iz(n, s), -1) for s in compat], -np.inf, 0)
            if n_anom[p]:
                add([(iA(p, l), 1), (iw(p, l), -1)], -np.inf, 0)
                add([(iA(p, l), 1), (ic(l), -1)], -np.inf, 0)
            if n_legit[p]:
                add([(iF(p, l), 1), (iw(p, l), -1), (ic(l), -1)], -1, np.inf)
    # false-block budget
    add([(iF(p, l), n_legit[p]) for p in range(P) if n_legit[p] for l in range(8)],
        -np.inf, budget * n_legit_total)

    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    cost = np.zeros(nvar)
    for p in range(P):
        for l in range(8):
            cost[iA(p, l)] = -n_anom[p]                  # maximise anomalies caught
            cost[iF(p, l)] = 1e-3 * n_legit[p]           # tie-break: fewer false blocks
    integrality = np.zeros(nvar)
    integrality[: 7 * S1 + 8] = 1
    ub = np.ones(nvar)
    ub[[iA(p, l) for p in range(P) for l in range(8) if not n_anom[p]]] = 0
    ub[[iF(p, l) for p in range(P) for l in range(8) if not n_legit[p]]] = 0

    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(np.zeros(nvar), ub),
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": 1e-4})
    if res.x is None:
        raise RuntimeError(f"MIO found no solution ({P} cells, {S} splits, {nvar} vars, "
                           f"{r} constraints): {res.message}")

    nodes = []
    for n in range(7):
        s = int(np.argmax(res.x[n * S1:(n + 1) * S1]))
        nodes.append((0, 255) if s == always_left else splits[s])
    leaves = sum(1 << l for l in range(8) if res.x[ic(l)] > 0.5)
    info = {"status": res.message, "gap": getattr(res, "mip_gap", None),
            "cells": P, "splits": S, "vars": nvar, "constraints": r,
            "caught": -float(res.fun)}
    return Tree(nodes, leaves), info
