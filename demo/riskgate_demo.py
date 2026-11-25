"""
RiskGate-ML live demo for the Tiny Tapeout demo board (MicroPython on RP2040).

Copy this file and tree_config.py to the board, then:
    import riskgate_demo; riskgate_demo.run()

The chip runs in manual-time mode, so the replay is deterministic: you get
exactly the decisions the Python model and the cocotb tests predict, no matter
how slowly MicroPython toggles the pins.  The 7-segment display shows each
decision (A = accepted, P/5/L/r/H = hard rule, d = ML detector, dot = ML flag).

The pin access below uses the Tiny Tapeout MicroPython SDK (ttboard).  If your
SDK version names things differently, only the four HAL functions need edits.
"""

import time

from ttboard.demoboard import DemoBoard

from tree_config import CONFIG, LEAVES, NODES

# ---------------------------------------------------------------- HAL ----
tt = DemoBoard.get()


def hal_init():
    tt.shuttle.tt_um_riskgate_ml.enable()
    tt.clock_project_PWM(25_000_000)
    tt.uio_oe_pico.value = 0b0011_1111   # RP2040 drives ADDR, WR, KILL
    tt.reset_project(True)
    time.sleep_ms(1)
    tt.reset_project(False)


def hal_set(ui, uio):
    tt.ui_in.value = ui
    tt.uio_in.value = uio


def hal_read():
    return int(tt.uo_out.value)


# ----------------------------------------------------- register access ----
WR = 0x10
CTRL_DISP, CTRL_ML_EN, CTRL_ENFORCE, CTRL_MANUAL = 1, 2, 4, 16
CTRL_RUN = CTRL_ML_EN | CTRL_ENFORCE | CTRL_MANUAL


def write(addr, data):
    hal_set(data, addr)
    hal_set(data, addr | WR)
    hal_set(data, addr)          # each pin update takes >> 3 chip clocks


def read(addr):
    hal_set(0, addr)
    return hal_read()


def configure():
    for addr, key in [(4, "ref"), (5, "band"), (6, "max_qty"), (7, "pos_lim"),
                      (8, "rate_max"), (9, "rate_win")]:
        write(addr, CONFIG[key])
    for n, (f, thr) in enumerate(NODES):
        write(0xB, (n << 4) | f)
        write(0xC, thr)
    write(0xD, LEAVES)
    write(0x0, CTRL_RUN)
    write(0xF, 0x0B)             # clear position, window, status


def ticks(n):
    for _ in range(n):
        write(0xF, 0x04)


REASONS = ["KILL", "PRICE BAND", "SIZE", "POSITION", "RATE", "ML DETECTOR"]


def order(side, price, qty, label=""):
    write(0x1, price)
    write(0x2, qty)
    write(0x3, side)
    s = read(0x0)
    leaf = read(0x8)
    why = [REASONS[b] for b in range(6) if s >> b & 1]
    verdict = "ACCEPT" if s & 0x40 else "REJECT"
    print(f"{label:<12} {'SELL' if side else 'BUY ':<4} {qty:>3} @ {price:<3} -> {verdict:<6} "
          f"{', '.join(why):<24} leaf={leaf}")
    write(0x0, CTRL_RUN | CTRL_DISP)   # show it on the 7-segment display
    time.sleep_ms(250)
    write(0x0, CTRL_RUN)
    return s


# ----------------------------------------------------------- scenario ----
def run():
    hal_init()
    configure()
    ref = CONFIG["ref"]
    print("--- normal trading")
    for side, dp, qty, gap in [(0, 1, 30, 12), (1, -2, 45, 9), (0, 0, 25, 15), (1, 3, 60, 11)]:
        ticks(gap)
        order(side, ref + dp, qty, "normal")
    print("--- runaway algo (Knight Capital style): small, in-band, relentless")
    for _ in range(16):
        ticks(1)
        order(0, ref + 1, 40, "runaway")
    ticks(60)
    print("--- fat-finger near miss: under every hard limit, still wrong")
    order(1, ref - 14, 190, "fat-finger")
    ticks(60)
    print("--- quote stuffing: tiny orders, flat out")
    for k in range(8):
        ticks(1)
        order(k % 2, ref, 2, "stuffing")
    pos = read(1) | (read(2) << 8)
    print(f"final position: {pos - 0x10000 if pos & 0x8000 else pos} shares")
