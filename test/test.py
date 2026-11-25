# SPDX-License-Identifier: Apache-2.0
"""
cocotb tests for RiskGate-ML.

Only top-level pins are used, so the same tests run on RTL and on the
gate-level netlist produced by the Tiny Tapeout GDS flow.  Every check
compares the chip against the bit-exact Python golden model in ../ml.
"""

import csv
import json
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))
from riskgate_model import (ACCEPT, KILL, ML, POS, PRICE, RATE, SIZE,  # noqa: E402
                            Config, RiskGateModel, Tree)

# register map (see src/project.v)
A_CTRL, A_PRICE, A_QTY, A_SUBMIT = 0x0, 0x1, 0x2, 0x3
A_REF, A_BAND, A_MAXQTY, A_POSLIM = 0x4, 0x5, 0x6, 0x7
A_RATEMAX, A_RATEWIN, A_TIMEBASE = 0x8, 0x9, 0xA
A_NODESEL, A_NODETHR, A_LEAVES, A_SIDE, A_CMD = 0xB, 0xC, 0xD, 0xE, 0xF
R_STATUS, R_POSLO, R_POSHI = 0x0, 0x1, 0x2
R_F0, R_LEAF, R_WINTICKS, R_F5, R_ID = 0x3, 0x8, 0x9, 0xA, 0xF
R_FEATS = [R_F0 + k for k in range(5)] + [R_F5]

CTRL_DISP, CTRL_ML_EN, CTRL_ENFORCE, CTRL_KILL, CTRL_MANUAL = 1, 2, 4, 8, 16
CMD_CLR_POS, CMD_CLR_WIN, CMD_STEP, CMD_CLR_STAT = 1, 2, 4, 8
WR, KILL_PIN = 0x10, 0x20


class Host:
    """Drives the chip the way the demo board's RP2040 would."""

    def __init__(self, dut):
        self.dut = dut
        self.kill = 0

    async def write(self, addr, data):
        d = self.dut
        d.ui_in.value = data & 0xFF
        d.uio_in.value = addr | self.kill
        await ClockCycles(d.clk, 1)
        d.uio_in.value = addr | WR | self.kill
        await ClockCycles(d.clk, 4)
        d.uio_in.value = addr | self.kill
        await ClockCycles(d.clk, 3)

    async def read(self, addr):
        self.dut.uio_in.value = addr | self.kill
        await ClockCycles(self.dut.clk, 1)
        return int(self.dut.uo_out.value)

    async def set_kill_pin(self, on):
        self.kill = KILL_PIN if on else 0
        self.dut.uio_in.value = self.kill
        await ClockCycles(self.dut.clk, 3)

    async def ticks(self, n):
        for _ in range(n):
            await self.write(A_CMD, CMD_STEP)

    async def configure(self, cfg: Config, manual=True, disp=False):
        for a, v in [(A_REF, cfg.ref), (A_BAND, cfg.band), (A_MAXQTY, cfg.max_qty),
                     (A_POSLIM, cfg.pos_lim), (A_RATEMAX, cfg.rate_max),
                     (A_RATEWIN, cfg.rate_win)]:
            await self.write(a, v)
        await self.load_tree(cfg.tree)
        ctrl = ((CTRL_DISP if disp else 0) | (CTRL_ML_EN if cfg.ml_en else 0) |
                (CTRL_ENFORCE if cfg.ml_enforce else 0) | (CTRL_KILL if cfg.kill else 0) |
                (CTRL_MANUAL if manual else 0))
        await self.write(A_CTRL, ctrl)
        await self.write(A_CMD, CMD_CLR_POS | CMD_CLR_WIN | CMD_CLR_STAT)

    async def load_tree(self, tree: Tree):
        for n, (f, thr) in enumerate(tree.nodes):
            await self.write(A_NODESEL, (n << 4) | f)
            await self.write(A_NODETHR, thr)
        await self.write(A_LEAVES, tree.leaves)

    async def order(self, side, price, qty):
        await self.write(A_PRICE, price)
        await self.write(A_QTY, qty)
        await self.write(A_SUBMIT, side)
        return await self.read(R_STATUS), await self.read(R_LEAF)

    async def position(self):
        lo, hi = await self.read(R_POSLO), await self.read(R_POSHI)
        v = (hi << 8) | lo
        return v - 0x10000 if v & 0x8000 else v


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 40, unit="ns").start())  # 25 MHz
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    return Host(dut)


async def check(host, model, side, price, qty, msg=""):
    status, leaf = await host.order(side, price, qty)
    exp = model.submit(side, price, qty)
    assert status == exp["status"], (
        f"{msg} order side={side} price={price} qty={qty}: "
        f"status 0x{status:02X} != expected 0x{exp['status']:02X} (fails={exp['fails']})")
    assert leaf == exp["leaf"], f"{msg} leaf {leaf} != {exp['leaf']}"
    return status


def bit(status, b):
    return (status >> b) & 1


@cocotb.test()
async def test_reset_and_id(dut):
    h = await start(dut)
    assert await h.read(R_ID) == 0x52
    assert await h.read(R_STATUS) == 0
    assert await h.position() == 0
    assert await h.read(R_F0 + 4) == 255, "idle counter should start saturated"
    assert int(dut.uio_oe.value) == 0xC0
    assert int(dut.uio_out.value) & 0xC0 == 0, "LEDs off before first decision"


@cocotb.test()
async def test_price_band(dut):
    h = await start(dut)
    cfg = Config(ref=100, band=5)
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    assert bit(await check(h, m, 0, 105, 10), ACCEPT)
    s = await check(h, m, 0, 106, 10)
    assert bit(s, PRICE) and not bit(s, ACCEPT)
    s = await check(h, m, 1, 94, 10)
    assert bit(s, PRICE)
    assert bit(await check(h, m, 1, 95, 10), ACCEPT)


@cocotb.test()
async def test_size(dut):
    h = await start(dut)
    cfg = Config(max_qty=50)
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    assert bit(await check(h, m, 0, 128, 50), ACCEPT)
    assert bit(await check(h, m, 0, 128, 51), SIZE)
    assert bit(await check(h, m, 0, 128, 0), SIZE)


@cocotb.test()
async def test_position_limit(dut):
    h = await start(dut)
    cfg = Config(pos_lim=2)   # 16 shares
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    assert bit(await check(h, m, 0, 128, 10), ACCEPT)
    assert bit(await check(h, m, 0, 128, 10), POS)       # would be +20
    assert bit(await check(h, m, 1, 128, 25), ACCEPT)    # -> -15
    assert await h.position() == -15
    assert bit(await check(h, m, 1, 128, 2), POS)        # would be -17
    await h.write(A_CMD, CMD_CLR_POS)
    m.pos = 0
    assert await h.position() == 0


@cocotb.test()
async def test_rate_window(dut):
    h = await start(dut)
    cfg = Config(rate_max=3, rate_win=4)   # 3 attempts per 5 ticks
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    for _ in range(3):
        assert bit(await check(h, m, 0, 128, 1), ACCEPT)
    assert bit(await check(h, m, 1, 128, 1), RATE)
    await h.ticks(4)
    m.tick(4)
    assert await h.read(R_WINTICKS) == 4
    assert bit(await check(h, m, 1, 128, 1), RATE), "window not over yet"
    await h.ticks(1)
    m.tick(1)
    assert await h.read(R_WINTICKS) == 0
    assert bit(await check(h, m, 1, 128, 1), ACCEPT), "new window"


@cocotb.test()
async def test_kill_switch(dut):
    h = await start(dut)
    cfg = Config()
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    await h.set_kill_pin(True)
    m.cfg.kill = True
    s = await check(h, m, 0, 128, 10, "kill pin")
    assert bit(s, KILL) and not bit(s, ACCEPT)
    await h.set_kill_pin(False)
    m.cfg.kill = False
    assert bit(await check(h, m, 0, 128, 10), ACCEPT)
    await h.write(A_CTRL, CTRL_KILL | CTRL_MANUAL)   # software kill
    m.cfg.kill = True
    assert bit(await check(h, m, 0, 128, 10, "soft kill"), KILL)


@cocotb.test()
async def test_tree_shadow_and_enforce(dut):
    h = await start(dut)
    # one-question tree: qty > 40 -> anomalous (leaf 4 = right, left, left)
    tree = Tree([(1, 40)] + [(0, 255)] * 6, leaves=1 << 4)
    cfg = Config(ml_en=True, ml_enforce=False, tree=tree)
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    s = await check(h, m, 0, 128, 50, "shadow")
    assert bit(s, ML) and bit(s, ACCEPT), "shadow mode flags but does not block"
    assert await h.read(R_LEAF) == 4
    assert not bit(await check(h, m, 0, 128, 40), ML)

    await h.write(A_CTRL, CTRL_ML_EN | CTRL_ENFORCE | CTRL_MANUAL)
    m.cfg.ml_enforce = True
    s = await check(h, m, 1, 128, 50, "enforce")
    assert bit(s, ML) and not bit(s, ACCEPT)
    assert s & 0x1F == 0, "no hard rule failed - ML alone blocked it"


@cocotb.test()
async def test_feature_preview(dut):
    h = await start(dut)
    cfg = Config()
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    await check(h, m, 0, 128, 100)
    await h.ticks(7)
    m.tick(7)
    await h.write(A_PRICE, 120)
    await h.write(A_QTY, 33)
    await h.write(A_SIDE, 1)
    feats = [await h.read(r) for r in R_FEATS]
    exp, _ = m.features(1, 120, 33)
    assert feats == exp, f"features {feats} != {exp}"


@cocotb.test()
async def test_side_streak(dut):
    h = await start(dut)
    cfg = Config(rate_max=255, pos_lim=255, max_qty=255)
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    await h.write(A_SIDE, 0)
    assert await h.read(R_F5) == 1, "first order ever: streak 1"
    for _ in range(3):
        await check(h, m, 0, 128, 1)
    await h.write(A_SIDE, 0)
    assert await h.read(R_F5) == 4
    await h.write(A_SIDE, 1)
    assert await h.read(R_F5) == 1, "side change restarts the streak"
    for _ in range(260):           # rejected orders count too; must saturate at 255
        await check(h, m, 1, 128, 0)
    await h.write(A_SIDE, 1)
    assert await h.read(R_F5) == 255
    await h.write(A_CMD, CMD_CLR_WIN)
    m.streak = 0
    assert await h.read(R_F5) == 1, "CMD clear window also clears the streak"


@cocotb.test()
async def test_feature_saturation(dut):
    """Position feature must saturate at 255 when |position after| >= 2048."""
    h = await start(dut)
    cfg = Config(pos_lim=255, max_qty=255, rate_max=255)
    m = RiskGateModel(cfg)
    await h.configure(cfg)
    for _ in range(8):
        assert bit(await check(h, m, 0, 128, 250), ACCEPT)   # -> +2000
    assert await h.position() == 2000
    await h.write(A_QTY, 100)
    await h.write(A_SIDE, 0)
    assert await h.read(R_F0 + 2) == 255, "|2100|/8 saturates"
    await h.write(A_SIDE, 1)
    assert await h.read(R_F0 + 2) == 1900 >> 3
    assert bit(await check(h, m, 0, 128, 100), POS)


@cocotb.test()
async def test_display_and_leds(dut):
    h = await start(dut)
    cfg = Config(band=5)
    await h.configure(cfg, disp=True)
    assert int(dut.uo_out.value) == 0x40, "'-' before first order"
    await h.order(0, 128, 10)
    assert int(dut.uo_out.value) == 0x77, "'A' accepted"
    assert int(dut.uio_out.value) & 0xC0 == 0x40, "ACCEPT LED"
    await h.order(0, 140, 10)
    assert int(dut.uo_out.value) == 0x73, "'P' price band"
    assert int(dut.uio_out.value) & 0xC0 == 0x80, "REJECT LED"


@cocotb.test()
async def test_auto_timebase(dut):
    h = await start(dut)
    await h.write(A_TIMEBASE, 2)        # tick every 4 clocks, auto mode
    await h.write(A_CTRL, 0)
    await h.order(0, 100, 1)
    await ClockCycles(dut.clk, 200)
    idle = await h.read(R_F0 + 4)
    assert 45 <= idle <= 60, f"idle={idle} after ~210 clocks at 4 clocks/tick"


@cocotb.test()
async def test_replay_parity(dut):
    """Replay a held-out trading session; chip must match the model on every order."""
    art = os.path.join(HERE, "..", "ml", "artifacts")
    with open(os.path.join(art, "tree.json")) as fh:
        chip = json.load(fh)
    with open(os.path.join(art, "replay_session.csv")) as fh:
        orders = [{k: (int(v) if k in ("tick", "side", "price", "qty", "label") else v)
                   for k, v in r.items()} for r in csv.DictReader(fh)]
    limit = int(os.environ.get("REPLAY_N", "0")) or len(orders)
    orders = orders[:limit]

    cfg = Config(**{**chip["config"], "ml_en": True, "ml_enforce": True},
                 tree=Tree.from_dict(chip))
    h = await start(dut)
    m = RiskGateModel(cfg)
    await h.configure(cfg)

    t, stats = 0, {"accepted": 0, "rule": 0, "ml": 0, "anomalous_blocked": 0, "anomalous": 0}
    for i, o in enumerate(orders):
        await h.ticks(o["tick"] - t)
        m.tick(o["tick"] - t)
        t = o["tick"]
        s = await check(h, m, o["side"], o["price"], o["qty"], f"#{i}")
        if bit(s, ACCEPT):
            stats["accepted"] += 1
        elif s & 0x1F:
            stats["rule"] += 1
        else:
            stats["ml"] += 1
        stats["anomalous"] += o["label"]
        stats["anomalous_blocked"] += o["label"] and not bit(s, ACCEPT)
        if i % 50 == 0:
            assert await h.position() == m.pos
    dut._log.info(f"replayed {len(orders)} orders over {t} ticks, all bit-exact: {stats}")
