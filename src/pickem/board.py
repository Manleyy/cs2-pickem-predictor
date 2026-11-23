"""Draw the Pick'Em forecast as an image (output/board.png).

    python -m pickem board            # redraw from output/predictions.json

Teams are shown as name cards coloured by VRS region (no logos).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, S = 1600, 900, 2  # logical size; drawn at S x and scaled down for smooth edges

BG = (13, 17, 27)
PANEL = (22, 28, 42)
CARD = (31, 39, 57)
LINE = (44, 54, 76)
TEXT = (234, 238, 245)
MUTED = (139, 149, 168)
GOLD = (242, 184, 75)
GREEN = (62, 207, 142)
RED = (239, 91, 91)
BLUE = (88, 140, 255)
REGION = {"Europe": ("EU", (76, 141, 255)), "Americas": ("AM", (255, 138, 61)),
          "Asia": ("AS", (179, 107, 255))}

_FONT_CANDIDATES = {
    "bold": ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf"],
}
_cache: dict = {}


def font(size: int, weight: str = "bold"):
    key = (size, weight)
    if key not in _cache:
        for name in _FONT_CANDIDATES[weight]:
            try:
                _cache[key] = ImageFont.truetype(name, size * S)
                break
            except OSError:
                continue
        else:
            _cache[key] = ImageFont.load_default(size * S)
    return _cache[key]


class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x0, y0, x1, y1, fill, r=10, outline=None, width=1):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, fill=fill,
                                 outline=outline, width=width * S)

    def text(self, x, y, s, size=16, color=TEXT, weight="bold", anchor="la"):
        self.d.text((x * S, y * S), s, font=font(size, weight), fill=color, anchor=anchor)

    def width(self, s, size=16, weight="bold"):
        return self.d.textlength(s, font=font(size, weight)) / S

    def fit(self, s, max_w, size, weight="bold"):
        """Shrink the font until the text fits max_w."""
        while size > 9 and self.width(s, size, weight) > max_w:
            size -= 1
        return size

    def save(self, path):
        self.img.resize((W, H), Image.LANCZOS).save(path, optimize=True)


def _pct(p):
    return f"{100 * p:.0f}%"


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
    region = {x["name"]: (x.get("region") or "") for x in p["profiles"]}
    picks = p["picks"]
    slot_of = {t: s for s, ts in picks.items() for t in ts}
    c = Canvas()

    # ---------------------------------------------------------------- header
    title = p["tournament"].replace(" - ", " · ")
    c.text(40, 34, title.upper(), 30)
    c.text(40, 78, f"Pick'Em forecast  ·  data up to {p['model'].get('data_until', '?')}"
                   f"  ·  {p['n_sims']:,} simulated stages", 17, MUTED, "regular")
    c.rect(1250, 26, 1560, 104, PANEL, 14, outline=GREEN, width=2)
    c.text(1405, 40, f"{100 * p['p_coin']:.1f}%", 34, GREEN, anchor="ma")
    c.text(1405, 82, "CHANCE OF 5+ CORRECT", 13, MUTED, anchor="ma")

    # --------------------------------------------------------- round 1 panel
    c.rect(40, 130, 500, 860, PANEL, 16)
    c.text(64, 152, "ROUND 1", 18)
    c.text(64 + c.width("ROUND 1 ", 18), 155, "best of one · win chance", 14, MUTED, "regular")
    y = 196
    for m in p["round1"]:
        a, b, pa = m["team_a"], m["team_b"], m["p_a_bo1"]
        c.rect(60, y, 480, y + 72, CARD, 10)
        sa, sb = c.fit(a, 175, 16), c.fit(b, 175, 16)
        c.text(76, y + 12, a, sa, TEXT if pa >= 0.5 else MUTED)
        c.text(464, y + 12, b, sb, TEXT if pa < 0.5 else MUTED, anchor="ra")
        bx0, bx1, by = 76, 464, y + 44
        split = bx0 + (bx1 - bx0) * pa
        c.rect(bx0, by, bx1, by + 14, LINE, 7)
        c.rect(bx0, by, split, by + 14, BLUE if pa >= 0.5 else (90, 100, 125), 7)
        c.text(bx0, by + 16, "", 1)
        c.text(bx0 + 6, by - 1, _pct(pa), 12, TEXT, anchor="la")
        c.text(bx1 - 6, by - 1, _pct(1 - pa), 12, TEXT, anchor="ra")
        y += 82

    # ------------------------------------------------------------ the ticket
    c.rect(530, 130, 1090, 860, PANEL, 16)
    c.text(554, 152, "THE TICKET", 18)
    c.text(554 + c.width("THE TICKET ", 18), 155, "chance of each pick coming true", 14, MUTED,
           "regular")

    def card(x, y, w, team, prob, accent):
        c.rect(x, y, x + w, y + 64, CARD, 10)
        c.rect(x, y, x + 6, y + 64, accent, 3)
        code, col = REGION.get(region.get(team, ""), ("··", MUTED))
        c.rect(x + 18, y + 22, x + 52, y + 42, col, 6)
        c.text(x + 35, y + 25, code, 11, BG, anchor="ma")
        name_w = w - 64 - 74
        if c.width(team, 17) <= name_w:
            c.text(x + 64, y + 21, team, 17, TEXT)
        elif c.width(team, 15) <= name_w:
            c.text(x + 64, y + 23, team, 15, TEXT)
        else:  # wrap long names onto two lines instead of shrinking them
            words = team.split()
            cut = max(1, len(words) // 2)
            l1, l2 = " ".join(words[:cut]), " ".join(words[cut:])
            fs = c.fit(max(l1, l2, key=len), name_w, 15)
            c.text(x + 64, y + 12, l1, fs, TEXT)
            c.text(x + 64, y + 32, l2, fs, TEXT)
        c.text(x + w - 16, y + 18, _pct(prob), 22, accent, anchor="ra")

    def section(y, label, sub, accent, teams, probs, cols):
        c.text(554, y, label, 16, accent)
        c.text(554 + c.width(label + "  ", 16), y + 2, sub, 13, MUTED, "regular")
        w = (512 - (cols - 1) * 12) / cols
        rows = (len(teams) + cols - 1) // cols
        for i, (t, pr) in enumerate(zip(teams, probs)):
            r, k = divmod(i, cols)
            card(554 + k * (w + 12), y + 28 + r * 76, w, t, pr, accent)
        return y + 28 + rows * 76 + 18

    y = 196
    t30 = picks["3-0"]
    y = section(y, "3-0", "two teams to go undefeated", GOLD, t30, [rec[t]["3-0"] for t in t30], 2)
    adv = sorted(picks["advance"], key=lambda t: -(rec[t]["3-1"] + rec[t]["3-2"]))
    y = section(y, "ADVANCE", "six teams to finish 3-1 or 3-2", GREEN, adv,
                [rec[t]["3-1"] + rec[t]["3-2"] for t in adv], 2)
    t03 = picks["0-3"]
    y = section(y, "0-3", "two teams to go winless", RED, t03, [rec[t]["0-3"] for t in t03], 2)

    # distribution of correct picks
    dist = p.get("correct_distribution") or []
    if dist and y < 760:
        c.d.line([554 * S, (y - 4) * S, 1066 * S, (y - 4) * S], fill=LINE, width=S)
        c.text(554, y + 6, "HOW MANY PICKS COME TRUE", 14)
        c.text(1066, y + 8, "5+ earns the coin", 12, GREEN, "regular", anchor="ra")
        top, bottom = y + 34, 826
        bw = 512 / len(dist)
        mx = max(dist)
        for i, v in enumerate(dist):
            x0 = 554 + i * bw + 5
            h = (bottom - top - 18) * v / mx if mx else 0
            col = GREEN if i >= 5 else (90, 100, 125)
            if h > 1:
                c.rect(x0, bottom - h, x0 + bw - 10, bottom, col, 4)
            if v >= 0.005:
                c.text(x0 + (bw - 10) / 2, bottom - h - 16, _pct(v), 11, MUTED, "regular", anchor="ma")
            c.text(x0 + (bw - 10) / 2, bottom + 6, str(i), 12, TEXT, anchor="ma")

    # --------------------------------------------------- numbers + advancing
    c.rect(1120, 130, 1560, 860, PANEL, 16)
    c.text(1144, 152, "KEY NUMBERS", 18)
    rows = [("Expected correct picks", f"{p['expected_correct']:.2f} / 10")]
    bt = _backtest_summary(Path(out_png).parent)
    if bt:
        n, avg, five = bt
        rows += [(f"Backtest, {n} past Major stages", f"{avg:.2f} correct"),
                 ("Backtest stages with 5+ correct", f"{five} of {n}")]
    y = 190
    for k, v in rows:
        c.text(1144, y, k, 15, MUTED, "regular")
        c.text(1536, y, v, 15, TEXT, anchor="ra")
        y += 30
    c.d.line([1144 * S, (y + 4) * S, 1536 * S, (y + 4) * S], fill=LINE, width=S)
    y += 22
    c.text(1144, y, "CHANCE TO ADVANCE", 18)
    y += 36
    order = sorted(rec.values(), key=lambda r: -r["advance"])
    for r in order:
        t, pa = r["team"], r["advance"]
        col = {"3-0": GOLD, "advance": GREEN, "0-3": RED}.get(slot_of.get(t), (90, 100, 125))
        c.text(1144, y, t, c.fit(t, 150, 13), TEXT, "regular")
        bx0, bx1 = 1300, 1490
        c.rect(bx0, y + 2, bx1, y + 14, LINE, 6)
        c.rect(bx0, y + 2, bx0 + max(12, (bx1 - bx0) * pa), y + 14, col, 6)
        c.text(1536, y, _pct(pa), 13, TEXT, anchor="ra")
        y += 25
    y += 14
    lx = 1144
    for label, col in [("3-0 pick", GOLD), ("advance pick", GREEN), ("0-3 pick", RED),
                       ("not picked", (90, 100, 125))]:
        c.rect(lx, y + 3, lx + 12, y + 15, col, 3)
        c.text(lx + 18, y + 1, label, 12, MUTED, "regular")
        lx += 18 + c.width(label, 12, "regular") + 16

    # ---------------------------------------------------------------- footer
    x = 40
    for code, col in REGION.values():
        c.rect(x, 873, x + 30, 889, col, 5)
        c.text(x + 15, 875, code, 10, BG, anchor="ma")
        x += 36
    c.text(x + 4, 873, "VRS region: Europe, Americas, Asia", 13, MUTED, "regular")
    c.text(1560, 873, "github.com/Manleyy/cs2-pickem-predictor", 13, MUTED, "regular", anchor="ra")
    c.save(out_png)
