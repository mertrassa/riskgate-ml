"""
Globally optimal depth-3 tree for the chip, with a proof of optimality.

This solves the same problem as the mixed-integer program in mio_tree.py:

  maximise   anomalous orders blocked by the ML that the hard rules let through
  subject to legit orders blocked only by the ML  <=  budget
             a full depth-3 tree of univariate integer splits  x_f > t

but by decomposition instead of a generic MIO solver (whose LP relaxation is
too weak for an open-source solver at this size).

Lagrangian relaxation: move the budget into the objective with a price
lambda per false block.  For fixed lambda the objective is a sum over the 8
leaves of max(0, caught - lambda * false), so the best depth-2 subtree under
each root branch can be found independently, and every candidate split pair is
scored at once with matrix products.  Sweeping lambda gives
  * a sequence of optimal trees; we keep the best one inside the budget, and
  * the Lagrangian dual bound  min_lambda [V(lambda) + lambda * budget],
    an upper bound on what ANY depth-3 tree over these splits can catch.
The gap between the two is the certified optimality gap.
"""

import numpy as np

from riskgate_model import Tree


def candidate_splits(X, max_per_feature=48, extra=()):
    """All integer thresholds for low-cardinality features, quantiles otherwise."""
    splits = set(extra)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) <= max_per_feature:
            ts = vals[:-1]  # x > max is never true
        else:
            ts = np.unique(np.floor(np.quantile(X[:, f], np.linspace(0, 1, max_per_feature + 2)[1:-1])))
        splits.update((f, int(t)) for t in ts if 0 <= t < 255)
    return sorted(splits) + [(0, 255)]  # last one = "always left" (pruned node)


def _g(x):
    return np.maximum(x, 0.0)


def optimal_tree(X, y, n_legit_total, budget, extra=(), lambdas=None, max_per_feature=48):
    """X, y: orders the hard rules accepted.  Returns (Tree, info)."""
    splits = candidate_splits(X, max_per_feature, extra)
    S = len(splits)
    R = np.stack([X[:, f] > t for f, t in splits], axis=1)
    keys, inv = np.unique(R, axis=0, return_inverse=True)
    inv = inv.ravel()
    a = np.bincount(inv, weights=(y == 1), minlength=len(keys)).astype(np.float64)
    nl = np.bincount(inv, weights=(y == 0), minlength=len(keys)).astype(np.float64)
    Rf = keys.astype(np.float64)
    Lf = 1.0 - Rf
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.geomspace(0.02, 200, 140)])
    lam = np.asarray(lambdas, dtype=np.float64)
    L = len(lam)
    B = budget * n_legit_total

    root_val = np.full((S, L), -np.inf)
    choice = {}  # (r, side) -> (s*, t*, u*) per lambda

    for r in range(S):
        for side in (0, 1):
            rows = np.flatnonzero(Rf[:, r] if side else Lf[:, r])
            if len(rows) == 0:
                root_val[r] = np.where(np.isinf(root_val[r]), 0.0, root_val[r]) if side else 0.0
                choice[(r, side)] = (np.full(L, S - 1), np.full(L, S - 1), np.full(L, S - 1))
                continue
            Rs, Ls, ar, nr = Rf[rows], Lf[rows], a[rows], nl[rows]
            # counts[leaf][s, t]: node split s, child split t
            def pair(M1, M2, w):
                return M1.T @ (M2 * w[:, None])
            A = {k: pair(M1, M2, ar) for k, (M1, M2) in
                 {"LL": (Ls, Ls), "LR": (Ls, Rs), "RL": (Rs, Ls), "RR": (Rs, Rs)}.items()}
            N = {k: pair(M1, M2, nr) for k, (M1, M2) in
                 {"LL": (Ls, Ls), "LR": (Ls, Rs), "RL": (Rs, Ls), "RR": (Rs, Rs)}.items()}
            # value of the left child (split t on node's left) and right child (split u)
            lv = _g(A["LL"][..., None] - lam * N["LL"][..., None]) + _g(A["LR"][..., None] - lam * N["LR"][..., None])
            rv = _g(A["RL"][..., None] - lam * N["RL"][..., None]) + _g(A["RR"][..., None] - lam * N["RR"][..., None])
            t_best, u_best = lv.argmax(axis=1), rv.argmax(axis=1)      # (S, L)
            sub = lv.max(axis=1) + rv.max(axis=1)                       # (S, L)
            s_best = sub.argmax(axis=0)                                 # (L,)
            val = sub.max(axis=0)
            choice[(r, side)] = (s_best, t_best[s_best, np.arange(L)], u_best[s_best, np.arange(L)])
            root_val[r] = val if side == 0 else root_val[r] + val
        # root_val[r] currently holds side0 + side1

    V = root_val.max(axis=0)
    r_best = root_val.argmax(axis=0)

    def build(j):
        r = r_best[j]
        s0, t0, u0 = (c[j] for c in choice[(r, 0)])
        s1, t1, u1 = (c[j] for c in choice[(r, 1)])
        idx = [r, s0, s1, t0, u0, t1, u1]
        nodes = [splits[i] for i in idx]
        tree = Tree(nodes, 0)
        # optimal leaf labels for this lambda
        caught, false = np.zeros(8), np.zeros(8)
        for k, key in enumerate(keys):
            path, n = 0, 0
            for _ in range(3):
                b = int(key[idx[n]])
                path = (path << 1) | b
                n = 2 * n + 1 + b
            caught[path] += a[k]
            false[path] += nl[k]
        lab = (caught - lam[j] * false) > 0
        tree.leaves = int(sum(1 << l for l in range(8) if lab[l]))
        return tree, caught[lab].sum(), false[lab].sum()

    best, results = None, []
    for j in range(L):
        tree, c, f = build(j)
        results.append((lam[j], c, f))
        if f <= B and (best is None or c > best[1]):
            best = (tree, c, f, lam[j])
    dual = float(np.min(V + lam * B))
    tree, c, f, lj = best
    info = {"caught": float(c), "false_blocks": float(f), "budget": B, "lambda": float(lj),
            "upper_bound": dual, "gap": dual - float(c), "splits": S, "cells": len(keys)}
    return tree, info
