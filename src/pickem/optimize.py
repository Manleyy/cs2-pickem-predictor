"""Pick'Em optimiser.

Valve's Stage Pick'Em: 2 teams to go 3-0, 2 teams to go 0-3 and 6 teams to
advance (finish 3-1 or 3-2). 5 correct earns the coin upgrade. Maximising the
expected number of correct picks isn't the same as maximising P(>=5 correct),
because picks are correlated (if your 3-0 pick goes 3-1, it also wrecks the
advance slots). So the optimiser scores every candidate pick set against all
simulated tournaments and hill-climbs on P(>= threshold), tie-broken by the
expected number correct.
"""
from __future__ import annotations

import itertools

import numpy as np

SLOTS = ("3-0", "advance", "0-3")


def outcome_matrices(wins, losses, advance_counts_3_0=False):
    is30 = (wins == 3) & (losses == 0)
    is03 = (wins == 0) & (losses == 3)
    adv = (wins == 3) if advance_counts_3_0 else ((wins == 3) & (losses > 0))
    return {"3-0": is30, "advance": adv, "0-3": is03}


def score(picks: dict[str, list[int]], M: dict, threshold: int):
    correct = sum(M[s][:, picks[s]].sum(1) for s in SLOTS)
    return float((correct >= threshold).mean()), float(correct.mean()), correct


def optimise(wins, losses, sizes: dict[str, int], threshold=5, advance_counts_3_0=False,
             locked: dict[str, list[int]] | None = None, restarts=8, seed=0,
             show_progress=False):
    M = outcome_matrices(wins, losses, advance_counts_3_0)
    n = wins.shape[1]
    locked = {s: list(v) for s, v in (locked or {}).items()}
    marg = {s: M[s].mean(0) for s in SLOTS}
    rng = np.random.default_rng(seed)

    def initial(randomise):
        picks, used = {}, {t for v in locked.values() for t in v}
        for s in ("3-0", "0-3", "advance"):
            cur = list(locked.get(s, []))
            order = np.argsort(-(marg[s] + (rng.normal(0, 0.05, n) if randomise else 0)))
            for t in order:
                if len(cur) >= sizes[s]:
                    break
                if t not in used:
                    cur.append(int(t)); used.add(int(t))
            picks[s] = cur
        return picks

    def key(p):
        a, b, _ = score(p, M, threshold)
        return (a, b)

    from .progress import Progress

    best, best_key = None, (-1, -1)
    bar = Progress(restarts, "Optimising picks") if show_progress else None
    for r in range(restarts):
        if bar:
            bar.update()
        picks = initial(randomise=r > 0)
        cur = key(picks)
        improved = True
        while improved:
            improved = False
            used = {t for v in picks.values() for t in v}
            moves = []
            for s in SLOTS:  # swap a pick for an unused team
                for i, t in enumerate(picks[s]):
                    if t in locked.get(s, []):
                        continue
                    for u in range(n):
                        if u not in used:
                            moves.append((s, i, None, None, u))
            for s1, s2 in itertools.combinations(SLOTS, 2):  # exchange between slots
                for i, t in enumerate(picks[s1]):
                    for j, u in enumerate(picks[s2]):
                        if t in locked.get(s1, []) or u in locked.get(s2, []):
                            continue
                        moves.append((s1, i, s2, j, None))
            for s1, i, s2, j, u in moves:
                cand = {s: list(v) for s, v in picks.items()}
                if s2 is None:
                    cand[s1][i] = u
                else:
                    cand[s1][i], cand[s2][j] = cand[s2][j], cand[s1][i]
                k = key(cand)
                if k > cur:
                    picks, cur, improved = cand, k, True
                    break
        if cur > best_key:
            best, best_key = picks, cur
    if bar:
        bar.close()
    p_coin, exp_correct, dist = score(best, M, threshold)
    return best, p_coin, exp_correct, np.bincount(dist, minlength=11) / len(dist)
