"""
Train the depth-3 anomaly tree for the chip and measure what the ML layer adds
on top of the hard rules.

  1. CART baseline (scikit-learn), hyperparameters picked on validation
     sessions; quantised and checked bit-exact against scikit-learn.
  2. Globally optimal depth-3 tree for the real objective (optimal_tree.py),
     with the false-block budget as a hard constraint.  This is the chip tree.

  python generate_data.py      # once
  python train.py

Outputs (ml/artifacts/):
  tree.json            chip configuration: rule limits + tree nodes/leaves
  tree.txt             human-readable tree (explainability)
  metrics.json         rules-only vs rules+ML results on held-out sessions
  replay_session.csv   one held-out session, replayed by the cocotb tests
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from optimal_tree import optimal_tree
from riskgate_model import FEATURE_NAMES, Config, Tree, replay

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
MAX_FALSE_BLOCK = 0.01  # model-risk budget: ML may block at most 1% of legit orders the rules allow


def load_sessions(path):
    sessions = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("session", "tick", "side", "price", "qty", "label"):
                r[k] = int(r[k])
            sessions[r["session"]].append(r)
    return [sessions[k] for k in sorted(sessions)]


def featurise(sessions, cfg):
    """Features exactly as the chip computes them, with the ML layer off."""
    base = Config(**{**cfg.__dict__, "ml_en": False, "ml_enforce": False})
    X, y = [], []
    for s in sessions:
        for o, r in zip(s, replay(s, base)):
            X.append(r["features"])
            y.append(o["label"])
    return np.array(X, dtype=np.int64), np.array(y)


def sklearn_to_chip(clf):
    """Convert a sklearn tree (max_depth<=3) into the chip's full heap layout."""
    t = clf.tree_
    nodes = [(0, 255)] * 7
    leaves = 0

    def label(i):
        return int(np.argmax(t.value[i][0]))

    def fill(sk, heap, depth):
        nonlocal leaves
        if depth == 3:
            if label(sk):
                leaves |= 1 << (heap - 7)
            return
        if t.children_left[sk] == -1:          # early leaf: pad with always-left node
            nodes[heap] = (0, 255)
            fill(sk, 2 * heap + 1, depth + 1)
            fill(sk, 2 * heap + 2, depth + 1)
            return
        # integer features: x <= 7.5  <=>  x <= 7
        thr = int(np.floor(t.threshold[sk]))
        nodes[heap] = (int(t.feature[sk]), max(0, min(255, thr)))
        fill(t.children_left[sk], 2 * heap + 1, depth + 1)
        fill(t.children_right[sk], 2 * heap + 2, depth + 1)

    fill(0, 0, 0)
    return Tree(nodes, leaves)


def evaluate(sessions, cfg):
    """End-to-end replay; returns block rates and leak-before-block per episode."""
    blocked = defaultdict(lambda: [0, 0])   # scenario -> [blocked, total]
    leaked = []                              # anomalous orders accepted before first block
    caught_by_ml_only = 0
    legit_ml_only = 0                        # legit orders blocked by ML that rules allowed
    for s in sessions:
        res = replay(s, cfg)
        episodes = defaultdict(list)
        for o, r in zip(s, res):
            b = blocked[o["scenario"]]
            b[0] += int(not r["accept"])
            b[1] += 1
            if not r["accept"] and not r["rule_reject"]:
                if o["label"]:
                    caught_by_ml_only += 1
                else:
                    legit_ml_only += 1
            if o["episode"]:
                episodes[o["episode"]].append(r["accept"])
        for acc in episodes.values():
            n = 0
            for a in acc:
                if not a:
                    break
                n += 1
            leaked.append(n)
    anom = [v for k, v in blocked.items() if k != "normal"]
    return {
        "normal_false_block_rate": blocked["normal"][0] / blocked["normal"][1],
        "normal_ml_only_block_rate": legit_ml_only / blocked["normal"][1],
        "anomaly_block_rate": sum(b for b, _ in anom) / sum(n for _, n in anom),
        "per_scenario_block_rate": {k: v[0] / v[1] for k, v in sorted(blocked.items())},
        "mean_orders_leaked_before_first_block": float(np.mean(leaked)),
        "anomalous_orders_caught_only_by_ml": caught_by_ml_only,
        "episodes": len(leaked),
    }


def fit(X, y, hp):
    crit, leaf, w = hp
    return DecisionTreeClassifier(max_depth=3, criterion=crit, min_samples_leaf=leaf,
                                  class_weight={0: 1.0, 1: w}, random_state=0).fit(X, y)


GRID = [(c, l, w) for c in ("gini", "entropy") for l in (10, 25, 50, 100)
        for w in (0.5, 1, 1.5, 2, 3, 4, 6)]
OPT_BUDGET = 0.006


def featurise_with_rules(sessions, cfg):
    """Like featurise, plus whether the hard rules already rejected each order."""
    base = Config(**{**cfg.__dict__, "ml_en": False, "ml_enforce": False})
    X, y, rr = [], [], []
    for s in sessions:
        for o, r in zip(s, replay(s, base)):
            X.append(r["features"])
            y.append(o["label"])
            rr.append(r["rule_reject"])
    return np.array(X, dtype=np.int64), np.array(y), np.array(rr)


def describe(tree):
    """Human-readable view of a chip tree (heap order, go right if feature > threshold)."""
    lines = []

    def walk(n, depth, path):
        pad = "    " * depth
        if depth == 3:
            leaf = int(path, 2)
            lines.append(f"{pad}-> {'ANOMALY' if tree.leaves >> leaf & 1 else 'normal'} (leaf {leaf})")
            return
        f, t = tree.nodes[n]
        if (f, t) == (0, 255):
            lines.append(f"{pad}(no question: always left)")
        else:
            lines.append(f"{pad}{FEATURE_NAMES[f]} <= {t}?  yes:")
        walk(2 * n + 1, depth + 1, path + "0")
        if (f, t) != (0, 255):
            lines.append(f"{pad}{FEATURE_NAMES[f]} >  {t}:")
            walk(2 * n + 2, depth + 1, path + "1")

    walk(0, 0, "")
    return "\n".join(lines) + "\n"


def main():
    sessions = load_sessions(os.path.join(HERE, "data", "orders.csv"))
    n = len(sessions)
    train_s, val_s, test_s = sessions[: int(.6 * n)], sessions[int(.6 * n): int(.8 * n)], sessions[int(.8 * n):]
    cfg = Config()

    Xtr, ytr = featurise(train_s, cfg)

    # Model selection on validation sessions: highest anomaly block rate whose
    # ML-only false-block rate stays within budget.
    best = None
    for hp in GRID:
        tree = sklearn_to_chip(fit(Xtr, ytr, hp))
        m = evaluate(val_s, Config(**{**cfg.__dict__, "ml_en": True, "ml_enforce": True, "tree": tree}))
        if m["normal_ml_only_block_rate"] <= MAX_FALSE_BLOCK and (best is None or m["anomaly_block_rate"] > best[1]):
            best = (hp, m["anomaly_block_rate"], m["normal_ml_only_block_rate"])
    hp = best[0]
    print(f"selected {hp}: val anomaly block {best[1]:.3f}, ML-only false block {best[2]:.4f}")
    w = {"criterion": hp[0], "min_samples_leaf": hp[1], "anomaly_class_weight": hp[2]}

    Xtv, ytv = featurise(train_s + val_s, cfg)
    clf = fit(Xtv, ytv, hp)
    cart = sklearn_to_chip(clf)

    # Bit-exact check: integer chip tree == sklearn on every sample.
    Xall, _ = featurise(sessions, cfg)
    chip_pred = np.array([cart.predict(list(x))[1] for x in Xall])
    assert np.array_equal(chip_pred, clf.predict(Xall)), "quantised tree disagrees with sklearn!"
    print(f"bit-exact: CART chip tree matches sklearn on all {len(Xall)} orders")

    # Globally optimal depth-3 tree for the real objective (optimal_tree.py):
    # most anomalies caught that the rules miss, with ML-only false blocks
    # capped at OPT_BUDGET on the training data (a margin under the 1% limit,
    # because the rate observed on new data runs higher than on training data).
    Xr, yr, rr = featurise_with_rules(train_s + val_s, cfg)
    extra = [nd for nd in cart.nodes if nd != (0, 255)]   # CART's tree is in the search space
    tree, opt = optimal_tree(Xr[~rr], yr[~rr], int((yr == 0).sum()), OPT_BUDGET,
                             extra=extra, max_per_feature=32)
    cart_caught = int(sum(cart.predict(list(x))[1] for x, yy in zip(Xr[~rr], yr[~rr]) if yy))
    print(f"optimal tree: catches {opt['caught']:.0f} on train+val (CART {cart_caught}), "
          f"proven within {opt['gap']:.1f} orders of the best depth-3 tree over {opt['splits']} splits")

    rules = evaluate(test_s, cfg)
    ml_cart = evaluate(test_s, Config(**{**cfg.__dict__, "ml_en": True, "ml_enforce": True, "tree": cart}))
    ml = evaluate(test_s, Config(**{**cfg.__dict__, "ml_en": True, "ml_enforce": True, "tree": tree}))

    os.makedirs(ART, exist_ok=True)
    chip = {"config": {k: v for k, v in cfg.__dict__.items() if k != "tree"},
            "feature_names": FEATURE_NAMES, **tree.to_dict()}
    with open(os.path.join(ART, "tree.json"), "w") as fh:
        json.dump(chip, fh, indent=2)
    with open(os.path.join(ART, "tree.txt"), "w") as fh:
        fh.write("Chip tree (globally optimal, optimal_tree.py):\n")
        fh.write(describe(tree))
        fh.write("\nchip heap nodes (feature, threshold): %s\nleaves mask: 0x%02X\n"
                 % (tree.nodes, tree.leaves))
        fh.write("\nBaseline CART tree (scikit-learn):\n")
        fh.write(export_text(clf, feature_names=FEATURE_NAMES))
    metrics = {"cart_hyperparameters": w, "optimal_tree": opt, "opt_budget": OPT_BUDGET,
               "cart_caught_train_val": cart_caught, "test_sessions": len(test_s),
               "rules_only": rules, "rules_plus_cart": ml_cart, "rules_plus_ml": ml}
    with open(os.path.join(ART, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    demo_cfg = os.path.join(HERE, "..", "demo", "tree_config.py")
    os.makedirs(os.path.dirname(demo_cfg), exist_ok=True)
    with open(demo_cfg, "w") as fh:
        fh.write("# Generated by ml/train.py - do not edit.\n")
        fh.write(f"CONFIG = {chip['config']!r}\n")
        fh.write(f"NODES = {tree.nodes!r}\n")
        fh.write(f"LEAVES = 0x{tree.leaves:02X}\n")
    with open(os.path.join(ART, "replay_session.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(test_s[0][0].keys()))
        wr.writeheader()
        wr.writerows(test_s[0])

    print(describe(tree))
    print(f"{'held-out test':<36}{'rules only':>12}{'+ CART':>10}{'+ optimal':>11}")
    for k, label in [("anomaly_block_rate", "anomalous orders blocked"),
                     ("normal_false_block_rate", "legit orders blocked (false)"),
                     ("normal_ml_only_block_rate", "  ...of which blocked only by ML"),
                     ("mean_orders_leaked_before_first_block", "orders leaked before 1st block")]:
        print(f"{label:<36}{rules[k]:>12.3f}{ml_cart[k]:>10.3f}{ml[k]:>11.3f}")
    print("per-scenario block rate (rules / CART / optimal):")
    for k in ml["per_scenario_block_rate"]:
        print(f"  {k:<12} {rules['per_scenario_block_rate'][k]:.3f}  "
              f"{ml_cart['per_scenario_block_rate'][k]:.3f}  {ml['per_scenario_block_rate'][k]:.3f}")


if __name__ == "__main__":
    main()
