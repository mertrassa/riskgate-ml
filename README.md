# RiskGate-ML

**A pre-trade risk gateway on a chip: hard risk rules plus a machine-learning anomaly detector,
deciding every order in a single clock cycle (40 ns at 25 MHz).**

Built for [Tiny Tapeout](https://tinytapeout.com) (one 1×1 tile, sky130).

## The problem

Banks run automated trading algorithms that can send thousands of orders a second. When one
breaks, it can do damage faster than any human can react. Knight Capital lost **$440M in 45
minutes** in 2012 because a faulty algorithm faced no effective pre-trade check. Regulators now
require these checks (Canada: NI 23-103; US: SEC Rule 15c3-5). Because every check adds delay
to every order, banks put them in hardware.

Fixed rules only catch dangers someone thought of in advance. An order that is slightly too big,
slightly off-price and arrives slightly too fast passes every rule, even though the combination
is suspicious.

## The design

```
                 ┌───────────────────────── RiskGate-ML (1 clock cycle) ─────────────────────────┐
order  ───────►  │  feature extraction ──►  5 hard rules ──────────────┐                         │
(side,price,qty) │  |price-ref|, qty, pos,  kill · price band · size   ├──► reject if ANY ─────► │ accept / reject
                 │  rate, idle, streak      position · rate            │    rule fails,         │ + reason code
                 │            │                                         │    or ML flags (when   │ + tree leaf
                 │            └──────►  depth-3 decision tree ──────────┘    enforcing)          │
                 │                      (trained in Python, loaded at start-up)                  │
                 └───────────────────────────────────────────────────────────────────────────────┘
```

* **Rules and ML work together.** The ML can only *add* rejections, never approve an order that
  broke a hard rule.
* **Explainable.** Every rejection reports which rule failed, or which leaf of the tree fired.
  The whole model is 7 yes/no questions you can read (below).
* **Shadow mode.** The ML can report without blocking, the way banks validate a new model before
  trusting it (model risk management, e.g. OSFI Guideline E-23).
* **Retrainable after tape-out.** The tree is stored in registers, not wired into the silicon.
* **Bit-exact.** The chip computes its features exactly as the Python golden model does, and the
  cocotb tests replay a held-out trading session on the chip, order by order, against that model.

### The trained model

The chip's tree is the **globally optimal** depth-3 tree for this task, found by mixed-integer
optimization (details below):

```
side_streak <= 5?
├─ yes: idle_ticks <= 1?
│       ├─ yes: qty <= 5?        → yes: ANOMALY (quote stuffing: tiny orders, back-to-back)
│       │                        → no:  normal
│       └─ no:  price_dev <= 10? → yes: normal
│                                → no:  ANOMALY (fat-finger / momentum: far from reference)
└─ no:  idle_ticks <= 2?
        ├─ yes: qty <= 19?       → yes: normal
        │                        → no:  ANOMALY (runaway algo: same side, again and again, fast)
        └─ no:  price_dev <= 8?  → yes: normal
                                 → no:  ANOMALY (momentum: one side, walking the price)
```

### Results (12 held-out trading sessions, 38 injected incidents)

| | Rules only | Rules + CART tree | Rules + optimal tree (chip) |
|---|---:|---:|---:|
| Anomalous orders blocked | 43.8% | 85.9% | **87.6%** |
| Bad orders that got through before the first block, per incident | 7.5 | 2.4 | **2.2** |
| Legit orders blocked by the ML that the rules allowed | n/a | 0.8% | 0.8% (budget: 1%) |

| Scenario | Rules only | Rules + optimal tree |
|---|---:|---:|
| Fat-finger near miss | 28.6% | 71.4% |
| Momentum ignition | 23.7% | 71.1% |
| Quote stuffing | 50.6% | 100% |
| Runaway algorithm | 46.6% | 84.4% |

The fat-finger test set has only 14 orders, so its rate moves in steps of about 7 points.
The data is synthetic (see `ml/generate_data.py`). A natural next step is to train on real
order-book data.

### What limits accuracy: features, not model size

The first version had 5 features and blocked 72% of anomalous orders, but only 50% of runaway
algorithm orders. Experiments on the same data:

| Model | Anomalous blocked | Runaway | Fits on chip? |
|---|---:|---:|---|
| 3-level tree, 5 features | 72.6% | 50% | yes |
| 4-level tree, 5 features | 72.0% | 50% | no (twice the tree storage) |
| **3-level tree, + side-streak feature** | **86.5%** | **81%** | **yes, about 9 more flip-flops** |
| Boosted trees (hundreds), + side streak | 94.0% | 90% | no |

A bigger tree didn't help, because the model couldn't see the one thing that gives a runaway
algorithm away: it keeps sending orders on the same side. Adding an 8-bit same-side counter
fixed that and brings the tiny tree within 8 points of a large boosted model. These experiment
numbers score each order on its own, which is why the current model shows 72.6% here and 71.7%
when replayed end to end. The remaining gap to 100% is mostly a labelling limit: every order in
an incident is labelled anomalous, but the first few look exactly like normal trading.

### Optimal tree by mixed-integer optimization (MIO)

scikit-learn's CART builds a tree greedily, one split at a time, and optimises impurity, which
only stands in for the real goal. `ml/optimal_tree.py` instead finds the best depth-3 tree
for the actual objective, with the budget as a hard constraint:

* **maximise** anomalous orders blocked by the ML that the hard rules let through,
* **subject to** legit orders blocked only by the ML ≤ 0.6% of legit orders on training data
  (a margin under the 1% limit, since the rate on new data runs higher),
* over every full depth-3 tree of integer splits the silicon can store.

`ml/mio_tree.py` is the textbook mixed-integer program (after Bertsimas & Dunn, *Optimal
Classification Trees*). At full data size it has 115k variables, too many for the open-source
solver bundled with SciPy. `optimal_tree.py` solves the same problem by Lagrangian decomposition:
price each false block, and the 8 leaves separate so every candidate tree can be scored with
matrix products. Sweeping the price also yields a **proven upper bound**. On small problems, both
solvers were checked against each other. On the full training data:

| | CART | Optimal tree |
|---|---:|---:|
| Anomalous orders caught (train + validation) | 1,021 | **1,055** |
| ML-only false blocks (train + validation) | 0.79% | **0.59%** |
| Proven distance from the best possible depth-3 tree (123 candidate splits) | unknown | **0.5 orders** |

The optimal tree beats CART on both measures at once. On held-out data the gain is modest (85.9%
→ 87.6%). The more useful result is the proof: about 88% is the ceiling for any depth-3 tree here,
so the remaining gap to a large boosted model (94%) comes from the tree's size limit on the chip,
not from how it was trained.

## Repository layout

| Path | What |
|---|---|
| `src/project.v`, `src/rg_tree.v` | Verilog: risk gateway + tree engine |
| `ml/riskgate_model.py` | Bit-exact Python golden model of the chip |
| `ml/generate_data.py` | Synthetic order flow with injected incidents |
| `ml/train.py` | CART baseline → optimal tree → evaluate → export |
| `ml/optimal_tree.py` | Globally optimal depth-3 tree with a proven bound |
| `ml/mio_tree.py` | The same problem as a mixed-integer program (for a commercial solver) |
| `report/report.tex` | Step-by-step report on how the hardware and software work |
| `ml/artifacts/` | Trained tree, metrics, held-out replay session |
| `test/` | cocotb tests (RTL and gate level) |
| `demo/` | MicroPython demo for the Tiny Tapeout demo board |
| `docs/info.md` | Datasheet: register map, how to test |

## Running it

ML pipeline (Windows or Linux, needs numpy and scikit-learn):

```powershell
cd ml; python generate_data.py; python train.py
```

Simulation (Linux or WSL, needs iverilog and cocotb):

```bash
cd test && make -B          # 13 tests, including the full held-out session replay
```

## Submitting to Tiny Tapeout

1. Create a repo from the Tiny Tapeout Verilog template for your shuttle (GitHub → *Use this
   template*), then copy these files over the template's files.
2. Rename `tt_um_riskgate_ml` to something unique, such as `tt_um_<github-username>_riskgate`, in
   `info.yaml`, `src/project.v`, `test/tb.v` and `demo/riskgate_demo.py`.
3. Fill in `author` in `info.yaml`, enable GitHub Pages and Actions, then push.
4. Check the **gds** action: it reports the real cell utilisation and runs the gate-level tests.
   The design has about 240 flip-flops and should fit one tile. If it doesn't, the easiest cut
   is 6-bit thresholds; dropping the position feature (unused by the current tree) is another.
