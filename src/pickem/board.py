"""Draw the Pick'Em forecast as an image (output/board.png).

    python -m pickem board            # redraw from output/predictions.json

Top: every team ranked by chance of 3-0, advancing 3-1/3-2, 0-3 and advancing at all.
Bottom: the model's picks with the next most likely teams for each slot.
Teams carry a coloured [EU] / [AM] / [ASIA] tag for their VRS region.
Uses the bundled Barlow Condensed font (SIL Open Font License, see fonts/OFL.txt).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, S = 1600, 1500, 2  # logical size; drawn at S x and scaled down for smooth edges
FONTS = Path(__file__).parent / "fonts"

BG = (17, 19, 23)
TEXT = (242, 242, 242)
MUTED = (125, 131, 143)
RULE = (42, 46, 54)
PANEL = (27, 30, 37)
GOLD = (242, 179, 61)
GREEN = (76, 195, 138)
RED = (229, 83, 75)
GREY = (70, 75, 86)
REGION = {"Europe": (88, 140, 255), "Americas": (255, 138, 61), "Asia": (179, 107, 255)}
DISPLAY = {"GOSH": "GOSH (BC.Game)", "PVISION": "PVISION (PARIVISION)"}
_cache: dict = {}


def font(size: int, weight: str = "Bold"):
    key = (size, weight)
    if key not in _cache:
        try:
            _cache[key] = ImageFont.truetype(str(FONTS / f"BarlowCondensed-{weight}.ttf"), size * S)
        except OSError:
            _cache[key] = ImageFont.load_default(size * S)
    return _cache[key]


class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(self, x, y, s, size, color=TEXT, weight="Bold", anchor="ls", spacing=0):
        f = font(size, weight)
        if not spacing:
            self.d.text((x * S, y * S), s, font=f, fill=color, anchor=anchor)
            return
        cx = x * S  # letter-spaced label (left-anchored)
        for ch in s:
            self.d.text((cx, y * S), ch, font=f, fill=color, anchor="ls")
            cx += self.d.textlength(ch, font=f) + spacing * S

    def width(self, s, size, weight="Bold"):
        return self.d.textlength(s, font=font(size, weight)) / S

    def hline(self, x0, x1, y, color=RULE, w=1):
        self.d.rectangle([x0 * S, y * S, x1 * S, y * S + w * S - 1], fill=color)

    def bar(self, x0, x1, y, h, color):
        self.d.rectangle([x0 * S, y * S, x1 * S, (y + h) * S], fill=color)

    def rect(self, x0, y0, x1, y1, fill, r=12, outline=None, w=0):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], r * S, fill=fill,
                                 outline=outline, width=w * S)

    def save(self, path):
        self.img.resize((W, H), Image.LANCZOS).save(path, optimize=True)


def _backtest_summary(out_dir: Path):
    f = out_dir / "backtest_stages.csv"
    if not f.exists():
        return None
    import pandas as pd
    r = pd.read_csv(f)
    cur = r[r.version.str.startswith("Current")]
    if cur.empty:
        return None
    return len(cur), cur.correct.mean(), int((cur.correct >= 5).sum())


class Ctx:
    """Everything the drawing parts need, loaded once from predictions.json."""

    def __init__(self, predictions: Path):
        self.p = p = json.loads(Path(predictions).read_text(encoding="utf-8"))
        self.rec = {r["team"]: r for r in p["records"]}
        self.region = {x["name"]: x.get("region") or "" for x in p["profiles"]}
        self.picks = p["picks"]
        self.slot_of = {t: s for s, ts in p["picks"].items() for t in ts}

    def ring(self, t):
        return REGION.get(self.region.get(t, ""), MUTED)

    @staticmethod
    def name(t):
        return DISPLAY.get(t, t)


L, R = 70, 1530


def draw_header(c, x, title):
    event, _, stage = x.p["tournament"].partition(" - ")
    c.text(L, 78, event.upper(), 22, MUTED, "SemiBold", spacing=3)
    c.text(L, 162, f"{stage.upper()} {title}", 84)
    c.text(R, 150, f"{100 * x.p['p_coin']:.0f}%", 96, GREEN, anchor="rs")
    c.text(R, 184, "CHANCE OF 5+ CORRECT", 20, MUTED, "SemiBold", anchor="rs")
    c.hline(L, R, 214, TEXT, 2)


TAG = {"Europe": "EU", "Americas": "AM", "Asia": "ASIA"}


def tag(c, x, region, xx, y, size=15):
    """Draw a coloured [EU]-style region tag; returns its right edge."""
    t = f"[{TAG.get(region, '?')}]"
    c.text(xx, y, t, size, REGION.get(region, MUTED), "Bold")
    return xx + c.width(t, size, "Bold")


def draw_board(predictions: Path, out_png: Path):
    """All 16 teams ranked for 3-0 / advance / 0-3 / chance to advance, then the model picks."""
    global W, H
    W, H = 1600, 1700
    x = Ctx(predictions)
    rec, slot_of, name = x.rec, x.slot_of, x.name
    c = Canvas()
    draw_header(c, x, "PICK'EM")

    # ---- ticket: three boxes side by side at the bottom ----
    def next_best(key):
        rest = sorted((r for r in rec.values() if slot_of.get(r["team"]) != key), key=lambda r: -r[key])[:2]
        return [(r["team"], r[key]) for r in rest]

    def next_adv():
        rest = sorted((r for r in rec.values() if r["team"] not in slot_of),
                      key=lambda r: -(r["3-1"] + r["3-2"]))[:4]
        return [(r["team"], r["3-1"] + r["3-2"]) for r in rest]

    PAD, HEAD, ROW = 24, 60, 80
    BH = HEAD + 2 * ROW + 104

    def pick(cx, cy, cw, t, pr, col):
        nm = name(t).upper()
        size = 19
        while size > 14 and c.width(nm, size, "SemiBold") > cw - 10:
            size -= 1
        c.text(cx, cy + 22, nm, size, TEXT, "SemiBold")
        e = c.width(f"{100 * pr:.0f}%", 36) + 10
        c.text(cx, cy + 62, f"{100 * pr:.0f}%", 36, col)
        tag(c, x, x.region.get(t, ""), cx + e, cy + 60)

    def box(x0, x1, y0, label, sub, teams, probs, col, runners, ncol, run_w):
        c.rect(x0, y0, x1, y0 + BH, PANEL, 14)
        c.bar(x0, x0 + 5, y0 + 14, BH - 28, col)
        c.text(x0 + PAD, y0 + 42, label, 30, col)
        if sub:
            c.text(x0 + PAD + c.width(label, 30) + 10, y0 + 42, sub, 16, MUTED, "SemiBold")
        cw = (x1 - x0 - 2 * PAD) / ncol
        for i, (t, pr) in enumerate(zip(teams, probs)):
            rr, k = divmod(i, ncol)
            pick(x0 + PAD + k * cw, y0 + HEAD + rr * ROW, cw, t, pr, col)
        ny = y0 + HEAD + 2 * ROW + 6
        c.hline(x0 + PAD, x1 - PAD, ny, RULE)
        c.text(x0 + PAD, ny + 28, "NEXT MOST LIKELY", 13, MUTED, "SemiBold")
        for k, (t, pr) in enumerate(runners):
            blk, row = divmod(k, 2)  # two per column, extra columns to the right
            rx0 = x0 + PAD + blk * (run_w + 64)
            ry = ny + 56 + row * 26
            c.text(rx0, ry, name(t).upper(), 17, MUTED, "SemiBold")
            c.text(rx0 + run_w, ry, f"{100 * pr:.0f}%", 17, MUTED, "SemiBold", anchor="rs")

    def draw_ticket_row(y0):
        g = 32
        cw4 = (R - L - 3 * g) / 4
        t30 = x.picks["3-0"]
        box(L, L + cw4, y0, "3-0", "", t30, [rec[t]["3-0"] for t in t30], GOLD,
            next_best("3-0"), 1, cw4 - 2 * PAD)
        adv = sorted(x.picks["advance"], key=lambda t: -(rec[t]["3-1"] + rec[t]["3-2"]))
        box(L + cw4 + g, L + 3 * cw4 + 2 * g, y0, "ADVANCE", "3-1 / 3-2", adv,
            [rec[t]["3-1"] + rec[t]["3-2"] for t in adv], GREEN, next_adv(), 3, cw4 - 2 * PAD)
        t03 = x.picks["0-3"]
        box(R - cw4, R, y0, "0-3", "", t03, [rec[t]["0-3"] for t in t03], RED,
            next_best("0-3"), 1, cw4 - 2 * PAD)
        return y0 + BH

    y = 214
    # ---- all 16 teams ----
    top = y + 62
    c.text(L, top, "ALL 16 TEAMS", 22, MUTED, "SemiBold", spacing=3)
    c.text(R, top, "COLOURED = PICKED BY THE MODEL", 16, MUTED, "SemiBold", anchor="rs")
    top += 54
    gap = 32
    colw = (R - L - 3 * gap) / 4
    slot_col = {"3-0": GOLD, "advance": GREEN, "0-3": RED}

    def ranked(ix, title, sub, key_fn, col, is_picked, pick_col):
        x0 = L + ix * (colw + gap)
        x1 = x0 + colw
        c.text(x0, top, title, 28, col)
        if sub:
            c.text(x0 + c.width(title, 28) + 10, top, sub, 15, MUTED, "SemiBold")
        c.hline(x0, x1, top + 12, col, 2)
        rows = sorted(rec.values(), key=lambda r: -key_fn(r))
        mx = max(key_fn(r) for r in rows) or 1
        yy = top + 14
        bar0, bar1 = x1 - 96, x1 - 50
        for i, r in enumerate(rows, 1):
            t, v = r["team"], key_fn(r)
            picked = is_picked(t)
            pc = pick_col(t)
            yy += 33
            c.text(x0, yy, f"{i:02d}", 17, MUTED, "Medium")
            nx = tag(c, x, x.region.get(t, ""), x0 + 28, yy - 1) + 8
            nm = name(t).upper()
            size = 19
            while size > 13 and c.width(nm, size, "SemiBold") > bar0 - nx - 8:
                size -= 1
            c.text(nx, yy, nm, size, TEXT if picked else MUTED, "SemiBold")
            c.bar(bar0, bar1, yy - 10, 4, RULE)
            c.bar(bar0, bar0 + (bar1 - bar0) * v / mx, yy - 10, 4, pc if picked else GREY)
            c.text(x1, yy, f"{100 * v:.0f}%", 19, pc if picked else TEXT, "SemiBold", anchor="rs")
        return yy

    ranked(0, "3-0", "", lambda r: r["3-0"], GOLD,
           lambda t: slot_of.get(t) == "3-0", lambda t: GOLD)
    ranked(1, "ADVANCE", "3-1 / 3-2", lambda r: r["3-1"] + r["3-2"], GREEN,
           lambda t: slot_of.get(t) == "advance", lambda t: GREEN)
    ranked(2, "0-3", "", lambda r: r["0-3"], RED,
           lambda t: slot_of.get(t) == "0-3", lambda t: RED)
    yy = ranked(3, "CHANCE TO ADVANCE", "", lambda r: r["advance"], TEXT,
                lambda t: t in slot_of, lambda t: slot_col.get(slot_of.get(t), GREY))

    top = yy + 84
    c.hline(L, R, top - 52, RULE)
    c.text(L, top, "MODEL PICKS", 22, MUTED, "SemiBold", spacing=3)
    yy = draw_ticket_row(top + 28)

    # ---- footer with text tags ----
    fy = yy + 72
    H = fy + 29
    c.hline(L, R, fy - 33, RULE)
    xx = L
    for reg in REGION:
        xx = tag(c, x, reg, xx, fy, 16) + 6
        c.text(xx, fy, reg.upper(), 16, MUTED, "Medium")
        xx += c.width(reg.upper(), 16, "Medium") + 22
    p = x.p
    bits = [f"EXPECTED {p['expected_correct']:.1f} OF 10 CORRECT",
            f"{p['n_sims']:,} SIMULATED STAGES",
            f"DATA UP TO {p['model'].get('data_until', '?')}"]
    bt = _backtest_summary(Path(out_png).parent)
    if bt:
        bits.append(f"BACKTEST: 5+ IN {bt[2]} OF {bt[0]} PAST MAJOR STAGES")
    c.text(xx + 8, fy, "   /   ".join(bits), 16, MUTED, "Medium")
    c.text(R, fy, "GITHUB.COM/MANLEYY/CS2-PICKEM-PREDICTOR", 16, MUTED, "Medium", anchor="rs")
    c.img = c.img.crop((0, 0, W * S, H * S))
    c.save(out_png)


if __name__ == "__main__":
    import sys
    draw_board(Path(sys.argv[1]), Path(sys.argv[2]))
