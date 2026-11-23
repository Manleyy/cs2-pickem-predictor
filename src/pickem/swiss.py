"""Monte Carlo simulation of a Valve Major Swiss stage.

Rules modelled (Valve Major supplemental rules, 2025-26):
* 16 teams, 3 wins advance, 3 losses eliminate, no rematches.
* Round 1 is fixed (seed 1 v 9, 2 v 10, ... or the published pairings).
* Later rounds pair inside each W-L group. Teams are ordered by Buchholz
  difficulty (sum of current W-L differential of every opponent faced), then
  initial seed; highest plays lowest, falling back to the next-best pairing
  that avoids a rematch.
* Matches that can advance or eliminate a team are Bo3, the rest Bo1.

`played` lets you lock in real results as the stage happens.
"""
from __future__ import annotations

import numpy as np

# Valve's priority list for a group of six teams in rounds 4-5 (indices into the
# group sorted by Buchholz then seed). Table taken from
# github.com/Leclowndu93150/pickem-prediction-model (MIT), where it reproduces
# all 99 real pairings of the Cologne 2026 Major.
SIX_TEAM_PRIORITIES = (
    ((0, 5), (1, 4), (2, 3)), ((0, 5), (1, 3), (2, 4)), ((0, 4), (1, 5), (2, 3)), ((0, 4), (1, 3), (2, 5)),
    ((0, 3), (1, 5), (2, 4)), ((0, 3), (1, 4), (2, 5)), ((0, 5), (1, 2), (3, 4)), ((0, 4), (1, 2), (3, 5)),
    ((0, 2), (1, 5), (3, 4)), ((0, 2), (1, 4), (3, 5)), ((0, 3), (1, 2), (4, 5)), ((0, 2), (1, 3), (4, 5)),
    ((0, 1), (2, 5), (3, 4)), ((0, 1), (2, 4), (3, 5)), ((0, 1), (2, 3), (4, 5)),
)


def _pair_group(group: list[int], seen: set[frozenset]) -> list[tuple[int, int]] | None:
    """Top vs bottom, backtracking to avoid rematches."""
    if not group:
        return []
    top, rest = group[0], group[1:]
    for j in range(len(rest) - 1, -1, -1):
        opp = rest[j]
        if frozenset((top, opp)) in seen:
            continue
        sub = _pair_group(rest[:j] + rest[j + 1:], seen)
        if sub is not None:
            return [(top, opp)] + sub
    return None


class Swiss:
    def __init__(self, n_teams: int, p_bo1: np.ndarray, p_bo3: np.ndarray,
                 round1: list[tuple[int, int]], wins=3, losses=3, bo3_decider=True,
                 played: list[tuple[int, int]] | None = None):
        self.n, self.p1, self.p3 = n_teams, p_bo1, p_bo3
        self.round1, self.W, self.L, self.bo3_dec = round1, wins, losses, bo3_decider
        self.played = list(played or [])  # (winner, loser) in play order

    def pairings(self, w, l, opps, rnd_no=2):
        active = [t for t in range(self.n) if w[t] < self.W and l[t] < self.L]
        diff = w - l
        buch = np.array([sum(diff[o] for o in opps[t]) for t in range(self.n)])
        seen = {frozenset((t, o)) for t in range(self.n) for o in opps[t]}
        out = []
        for rec in sorted({(w[t], l[t]) for t in active}, key=lambda r: (-r[0], r[1])):
            g = sorted([t for t in active if (w[t], l[t]) == rec], key=lambda t: (-buch[t], t))
            pg = None
            if len(g) == 6 and rnd_no >= 4:
                for pat in SIX_TEAM_PRIORITIES:
                    if all(frozenset((g[i], g[j])) not in seen for i, j in pat):
                        pg = [(g[i], g[j]) for i, j in pat]
                        break
            if pg is None:
                pg = _pair_group(g, seen)
            if pg is None:  # extremely rare; accept a rematch rather than fail
                pg = [(g[i], g[-1 - i]) for i in range(len(g) // 2)]
            out += pg
        return out

    def run_once(self, rng) -> tuple[np.ndarray, np.ndarray]:
        w = np.zeros(self.n, int)
        l = np.zeros(self.n, int)
        opps = [[] for _ in range(self.n)]
        fixed = list(self.played)
        rnd = 0
        while True:
            pairs = self.round1 if rnd == 0 else self.pairings(w, l, opps, rnd + 1)
            if not pairs:
                break
            for a, b in pairs:
                # use a real result if it has been entered for this pair
                res = next((r for r in fixed if set(r) == {a, b}), None)
                if res is not None:
                    fixed.remove(res)
                    win = res[0]
                else:
                    decider = self.bo3_dec and (w[a] == self.W - 1 or l[a] == self.L - 1)
                    p = (self.p3 if decider else self.p1)[a, b]
                    win = a if rng.random() < p else b
                lose = b if win == a else a
                w[win] += 1
                l[lose] += 1
                opps[a].append(b)
                opps[b].append(a)
            rnd += 1
        return w, l

    def simulate(self, n_sims: int, seed: int = 0, show_progress: bool = False) -> np.ndarray:
        """Returns int array (n_sims, n_teams) of final wins (losses = implied).
        Codes: wins 3 -> advanced with 3-0/3-1/3-2 (use losses to tell)."""
        rng = np.random.default_rng(seed)
        wins = np.zeros((n_sims, self.n), np.int8)
        losses = np.zeros((n_sims, self.n), np.int8)
        bar = None
        if show_progress:
            from .progress import Progress
            bar = Progress(n_sims, "Simulating Swiss")
        for s in range(n_sims):
            wins[s], losses[s] = self.run_once(rng)
            if bar:
                bar.update()
        if bar:
            bar.close()
        return wins, losses


def record_probs(wins, losses, names) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for i, t in enumerate(names):
        w, l = wins[:, i], losses[:, i]
        row = {"team": t}
        for rec in ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]:
            a, b = map(int, rec.split("-"))
            row[rec] = float(((w == a) & (l == b)).mean())
        row["advance"] = float((w == 3).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("advance", ascending=False).reset_index(drop=True)
