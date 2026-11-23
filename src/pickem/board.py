"""Draw the Pick'Em forecast as an image (output/board.png).

    python -m pickem board            # redraw from output/predictions.json

Uses the bundled Barlow Condensed font (SIL Open Font License, see fonts/OFL.txt).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, S = 1600, 900, 2  # logical size; drawn at S x and scaled down for smooth edges
FONTS = Path(__file__).parent / "fonts"

BG = (17, 19, 23)
TEXT = (242, 242, 242)
MUTED = (125, 131, 143)
RULE = (42, 46, 54)
GOLD = (242, 179, 61)
GREEN = (76, 195, 138)
RED = (229, 83, 75)
GREY = (70, 75, 86)
REGION = {"Europe": "EU", "Americas": "AM", "Asia": "AS"}
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


def draw_board(predictions: Path, out_png: Path):
    p = json.loads(Path(predictions).read_text(encoding="utf-8"))
    rec = {r["team"]: r for r in p["records"]}
    region = {x["name"]: REGION.get(x.get("region") or "", "") for x in p["profiles"]}
    picks = p["picks"]
    slot_of = {t: s for s, ts in picks.items() for t in ts}
    c = Canvas()
    L, R = 70, 1530

    # ---------------------------------------------------------------- header
    event, _, stage = p["tournament"].partition(" - ")
    c.text(L, 78, event.upper(), 22, MUTED, "SemiBold", spacing=3)
    c.text(L, 162, f"{stage.upper()} PICK'EM", 84)
    c.text(R, 150, f"{100 * p['p_coin']:.0f}%", 96, GREEN, anchor="rs")
    c.text(R, 184, "CHANCE OF 5+ CORRECT", 20, MUTED, "SemiBold", anchor="rs")
    c.hline(L, R, 214, TEXT, 2)

    # ------------------------------------------------------- left: the ticket
    lx0, lx1 = L, 730
    y = 266

    def group(label, color, teams, probs):
        nonlocal y
        c.text(lx0, y, label, 28, color)
        c.text(lx1, y, "CHANCE", 15, MUTED, "SemiBold", anchor="rs")
        y += 12
        c.hline(lx0, lx1, y, color, 2)
        for t, pr in zip(teams, probs):
            y += 31
            c.text(lx0, y, t.upper(), 27, TEXT, "SemiBold")
            reg = region.get(t, "")
            if reg:
                c.text(lx0 + c.width(t.upper(), 27, "SemiBold") + 10, y, reg, 16, MUTED, "Medium")
            c.text(lx1, y, f"{100 * pr:.0f}%", 27, color, "SemiBold", anchor="rs")
            y += 9
            c.hline(lx0, lx1, y)
        y += 46

    t30 = picks["3-0"]
    group("3-0", GOLD, t30, [rec[t]["3-0"] for t in t30])
    adv = sorted(picks["advance"], key=lambda t: -(rec[t]["3-1"] + rec[t]["3-2"]))
    group("ADVANCE  3-1 / 3-2", GREEN, adv, [rec[t]["3-1"] + rec[t]["3-2"] for t in adv])
    t03 = picks["0-3"]
    group("0-3", RED, t03, [rec[t]["0-3"] for t in t03])

    # ------------------------------------------------ right: chance to advance
    rx0, rx1 = 830, R
    y = 268
    c.text(rx0, y, "CHANCE TO ADVANCE", 30)
    c.text(rx1, y, "ALL 16 TEAMS", 16, MUTED, "SemiBold", anchor="rs")
    y += 14
    c.hline(rx0, rx1, y, TEXT, 2)
    y += 8
    bar0, bar1 = rx0 + 290, rx1 - 70
    for i, r in enumerate(sorted(rec.values(), key=lambda r: -r["advance"]), 1):
        t, pa = r["team"], r["advance"]
        col = {"3-0": GOLD, "advance": GREEN, "0-3": RED}.get(slot_of.get(t), GREY)
        y += 33
        c.text(rx0, y, f"{i:02d}", 20, MUTED, "Medium")
        c.text(rx0 + 40, y, t.upper(), 22, TEXT if slot_of.get(t) else MUTED, "SemiBold")
        c.bar(bar0, bar1, y - 11, 4, RULE)
        c.bar(bar0, bar0 + (bar1 - bar0) * pa, y - 11, 4, col)
        c.text(rx1, y, f"{100 * pa:.0f}%", 22, TEXT, "SemiBold", anchor="rs")

    # ---------------------------------------------------------------- footer
    c.hline(L, R, 838, RULE)
    bits = [f"EXPECTED {p['expected_correct']:.1f} OF 10 CORRECT",
            f"{p['n_sims']:,} SIMULATED STAGES",
            f"DATA UP TO {p['model'].get('data_until', '?')}"]
    bt = _backtest_summary(Path(out_png).parent)
    if bt:
        n, avg, five = bt
        bits.append(f"BACKTEST: 5+ CORRECT IN {five} OF {n} PAST MAJOR STAGES")
    c.text(L, 870, "   /   ".join(bits), 17, MUTED, "Medium")
    c.text(R, 870, "GITHUB.COM/MANLEYY/CS2-PICKEM-PREDICTOR", 17, MUTED, "Medium", anchor="rs")
    c.save(out_png)
