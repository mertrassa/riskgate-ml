"""
Bit-exact Python golden model of the RiskGate-ML chip (src/project.v).

Used three ways:
  * generate_data.py / train.py compute ML features exactly as the chip does,
  * train.py evaluates the full rules + ML pipeline,
  * test/test.py checks the silicon (RTL or gate-level) order-by-order.

Time is modelled in ticks.  Orders arrive between ticks (the chip's manual
time mode, CMD.step, makes this exact).  Pure Python, no dependencies.
"""

from dataclasses import dataclass, field

FEATURE_NAMES = [
    "price_dev",    # f0 |price - ref|
    "qty",          # f1 order quantity
    "pos_after_8",  # f2 |net position after order| / 8, saturating at 255
    "win_count",    # f3 order attempts in current rate window (saturating)
    "idle_ticks",   # f4 ticks since previous order, saturating at 255
    "side_streak",  # f5 consecutive same-side orders including this one, saturating
]

# STATUS register bits
KILL, PRICE, SIZE, POS, RATE, ML, ACCEPT, TOGGLE = range(8)
REASONS = ["kill", "price", "size", "position", "rate", "ml"]


@dataclass
class Tree:
    """Full depth-3 tree in heap order: nodes[n] = (feature, threshold)."""
    nodes: list = field(default_factory=lambda: [(0, 255)] * 7)
    leaves: int = 0  # bit i = label of leaf i

    def predict(self, feats):
        n, path = 0, 0
        for _ in range(3):
            f, thr = self.nodes[n]
            b = 1 if (feats[f] if f < len(feats) else 0) > thr else 0
            path = (path << 1) | b
            n = 2 * n + 1 + b
        return path, (self.leaves >> path) & 1

    def to_dict(self):
        return {"nodes": [{"feature": f, "threshold": t} for f, t in self.nodes],
                "leaves_mask": self.leaves}

    @classmethod
    def from_dict(cls, d):
        return cls([(n["feature"], n["threshold"]) for n in d["nodes"]], d["leaves_mask"])


@dataclass
class Config:
    ref: int = 128
    band: int = 20
    max_qty: int = 200
    pos_lim: int = 125      # x8 shares -> 1000 shares
    rate_max: int = 12
    rate_win: int = 49      # window = 50 ticks
    ml_en: bool = False
    ml_enforce: bool = False
    kill: bool = False
    tree: Tree = field(default_factory=Tree)


def _sat8(x):
    return 255 if x > 255 else x


class RiskGateModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = 0
        self.win_ticks = 0
        self.win_count = 0
        self.idle = 255
        self.streak = 0       # 0 = no order yet
        self.last_side = 0
        self.toggle = 0

    # -- time -------------------------------------------------------------
    def tick(self, n=1):
        for _ in range(n):
            if self.idle != 255:
                self.idle += 1
            if self.win_ticks == self.cfg.rate_win:
                self.win_ticks = 0
                self.win_count = 0
            else:
                self.win_ticks = (self.win_ticks + 1) & 0xFF

    # -- features ---------------------------------------------------------
    def features(self, side, price, qty):
        pos_after = self.pos - qty if side else self.pos + qty
        same = self.streak != 0 and side == self.last_side
        feats = [
            abs(price - self.cfg.ref),
            qty,
            _sat8(abs(pos_after) >> 3),
            self.win_count,
            self.idle,
            _sat8(self.streak + 1) if same else 1,
        ]
        return feats, pos_after

    # -- order evaluation -------------------------------------------------
    def submit(self, side, price, qty):
        c = self.cfg
        feats, pos_after = self.features(side, price, qty)
        fails = {
            "kill": c.kill,
            "price": feats[0] > c.band,
            "size": qty == 0 or qty > c.max_qty,
            "position": abs(pos_after) > c.pos_lim * 8,
            "rate": self.win_count >= c.rate_max,
        }
        leaf, label = c.tree.predict(feats)
        ml_hit = bool(c.ml_en and label)
        reject = any(fails.values()) or (ml_hit and c.ml_enforce)
        accept = not reject

        self.idle = 0
        self.streak = feats[5]
        self.last_side = side
        if self.win_count != 255:      # counts every attempt, saturating
            self.win_count += 1
        if accept:
            self.pos = pos_after
        self.toggle ^= 1

        status = (int(fails["kill"]) << KILL | int(fails["price"]) << PRICE |
                  int(fails["size"]) << SIZE | int(fails["position"]) << POS |
                  int(fails["rate"]) << RATE | int(ml_hit) << ML |
                  int(accept) << ACCEPT | self.toggle << TOGGLE)
        return {
            "status": status,
            "accept": accept,
            "rule_reject": any(fails.values()),
            "ml_hit": ml_hit,
            "leaf": leaf,
            "features": feats,
            "fails": fails,
        }


def replay(orders, cfg, record=None):
    """Run a list of order dicts (tick, side, price, qty, ...) through the model.

    Orders must be sorted by tick with at most one order per tick boundary
    gap >= 0.  Returns a list of result dicts aligned with `orders`.
    """
    m = RiskGateModel(cfg)
    out, t = [], 0
    for o in orders:
        m.tick(o["tick"] - t)
        t = o["tick"]
        out.append(m.submit(o["side"], o["price"], o["qty"]))
    return out
