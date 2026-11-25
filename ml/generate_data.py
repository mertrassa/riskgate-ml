"""
Synthetic order-flow generator for training the RiskGate-ML anomaly tree.

Each *session* is one trading algorithm's order stream, starting from a reset
chip (flat position).  Normal flow is a mean-reverting market-making style
algo.  Anomalous episodes are injected and every order inside one is labelled
1.  Scenarios are modelled on real failure modes:

  runaway      - an algo stuck in a loop firing same-side orders (Knight
                 Capital, 2012).  Each order is small and inside the price
                 band, so individually it looks fine.
  fat_finger   - an oversized order that is still just under the size cap.
  momentum     - an aggressive same-side sequence walking the price away
                 from the reference (momentum ignition).
  stuffing     - a flood of tiny alternating orders (quote stuffing).

One tick ~ 1 ms on the demo board (TIMEBASE = 14 at 25 MHz gives ~0.66 ms).

Usage:  python generate_data.py [--sessions 60] [--seed 7]
"""

import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = 128
SESSION_TICKS = 3000


def _clip(x, lo, hi):
    return int(max(lo, min(hi, round(x))))


def normal_order(rng, pos):
    p_buy = float(np.clip(0.5 - pos / 1200.0, 0.1, 0.9))
    side = 0 if rng.random() < p_buy else 1
    price = _clip(REF + rng.normal(0, 3.5), 0, 255)
    qty = _clip(rng.lognormal(np.log(30), 0.6), 1, 180)
    return side, price, qty


def episode(rng, kind):
    """Return list of (gap, side, price, qty) for one anomalous episode."""
    out = []
    side = int(rng.integers(0, 2))
    sgn = 1 if side == 0 else -1  # buyers push price up, sellers down
    if kind == "runaway":
        for _ in range(int(rng.integers(15, 41))):
            out.append((int(rng.integers(1, 3)), side,
                        _clip(REF + sgn * abs(rng.normal(1, 2)), 0, 255),
                        int(rng.integers(20, 61))))
    elif kind == "fat_finger":
        for _ in range(int(rng.integers(1, 3))):
            out.append((int(rng.integers(3, 20)), side,
                        _clip(REF + sgn * rng.uniform(6, 20), 0, 255),
                        int(rng.integers(140, 201))))
    elif kind == "momentum":
        dev = rng.uniform(3, 6)
        for _ in range(int(rng.integers(8, 16))):
            dev += rng.uniform(0.8, 2.2)
            out.append((int(rng.integers(2, 6)), side,
                        _clip(REF + sgn * dev, 0, 255),
                        int(rng.integers(30, 81))))
    elif kind == "stuffing":
        for k in range(int(rng.integers(20, 51))):
            out.append((1, (side + k) % 2,
                        _clip(REF + rng.normal(0, 2), 0, 255),
                        int(rng.integers(1, 6))))
    return out


def gen_session(rng, sid):
    rows, t, pos = [], 0, 0
    kinds = ["runaway", "fat_finger", "momentum", "stuffing"]
    # schedule 2-4 episodes at random times
    starts = sorted(rng.choice(np.arange(200, SESSION_TICKS - 300), size=int(rng.integers(2, 5)),
                               replace=False).tolist())
    next_ep = 0
    while t < SESSION_TICKS:
        if next_ep < len(starts) and t >= starts[next_ep]:
            kind = kinds[int(rng.integers(0, len(kinds)))]
            for gap, side, price, qty in episode(rng, kind):
                t += gap
                rows.append(dict(session=sid, tick=t, side=side, price=price, qty=qty,
                                 label=1, scenario=kind, episode=f"{sid}-{next_ep}"))
            next_ep += 1
            continue
        # legit bursts (hedging / rebalancing) make the problem non-trivial
        if rng.random() < 0.04:
            burst = int(rng.integers(3, 7))
            gaps = rng.integers(1, 4, size=burst)
        else:
            burst, gaps = 1, [max(1, int(rng.geometric(1 / 12)))]
        for g in gaps:
            t += int(g)
            side, price, qty = normal_order(rng, pos)
            pos += -qty if side else qty
            pos = int(np.clip(pos, -1000, 1000))
            rows.append(dict(session=sid, tick=t, side=side, price=price, qty=qty,
                             label=0, scenario="normal", episode=""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "orders.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for sid in range(args.sessions):
        rows.extend(gen_session(rng, sid))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_anom = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} orders ({n_anom} anomalous, {n_anom / len(rows):.1%}) "
          f"in {args.sessions} sessions -> {args.out}")


if __name__ == "__main__":
    main()
