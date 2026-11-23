"""Backtest on past Major Swiss stages (data/backtest/majors.yaml).

For each stage, and for each model version:
  1. fit the model only on matches played before the stage started
  2. simulate the stage and choose Pick'Em picks exactly like `predict`
  3. score the picks against the real final records
  4. score per-match accuracy on the stage's real matches (from the VRS data)

A "pick by seed" baseline is included: seeds 1-2 to go 3-0, 15-16 to go 0-3,
seeds 3-8 to advance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model import MatchModel, bo3_from_map, map_from_bo3, stretch
from .optimize import optimise
from .swiss import Swiss


def _score_picks(picks: dict[str, list[str]], records: dict[str, str], adv_30=False) -> int:
    ok = 0
    for t in picks["3-0"]:
        ok += records[t] == "3-0"
    for t in picks["0-3"]:
        ok += records[t] == "0-3"
    for t in picks["advance"]:
        ok += records[t] in (("3-1", "3-2", "3-0") if adv_30 else ("3-1", "3-2"))
    return ok


def run(cfg, fb, df: pd.DataFrame, matches: pd.DataFrame, names, stages_file: Path,
        n_sims: int, versions: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .progress import Progress

    stages = yaml.safe_load(Path(stages_file).read_text(encoding="utf-8"))
    pc, sw, mc = cfg["pickem"], cfg["swiss"], cfg["model"]
    sizes = {"3-0": pc["n_3_0"], "0-3": pc["n_0_3"], "advance": pc["n_advance"]}
    rows, match_rows = [], []
    bar = Progress(len(stages) * len(versions), "Backtesting")
    for st in stages:
        when = pd.Timestamp(st["date"])
        shown = [p[0] for p in st["round1"]] + [p[1] for p in st["round1"]]  # seed order
        canon = [names(t) for t in shown]
        idx = {t: i for i, t in enumerate(shown)}
        round1 = [(idx[a], idx[b]) for a, b in st["round1"]]
        records = st["records"]
        n = len(shown)
        prof = [fb.team_profile(c, when) for c in canon]
        pairs = [[fb.pair(prof[i], prof[j]) if i != j else None for j in range(n)] for i in range(n)]

        # the stage's real matches (both teams in the stage, inside its window)
        end = when + pd.Timedelta(days=5 if st.get("all_bo3") else 4)
        cset = set(canon)
        real = matches[(matches.date >= when) & (matches.date < end) &
                       matches.winner.isin(cset) & matches.loser.isin(cset)]
        c2i = {c: i for i, c in enumerate(canon)}

        # baselines: pick straight down a ranking (top 2 -> 3-0, next 6 -> advance, bottom 2 -> 0-3)
        def by_order(order):
            return {"3-0": order[:2], "advance": order[2:8], "0-3": order[-2:]}

        vrs_order = sorted(shown, key=lambda t: -fb.vrs_points(names(t), when)[0])
        hltv_order = sorted(shown, key=lambda t: (np.nan_to_num(fb.hltv_rank(names(t), when), nan=999),
                                                  -fb.vrs_points(names(t), when)[0]))
        baselines = [("Pick by seed (listing order)", shown), ("Pick by VRS points", vrs_order)]
        if fb.hltv_dates:  # only with HLTV data (python -m pickem hltv-fetch)
            baselines.append(("Pick by HLTV rank", hltv_order))
        for label, order in baselines:
            pk = by_order(order)
            rows.append(dict(stage=st["name"], version=label,
                             correct=_score_picks(pk, records, pc["advance_counts_3_0"]),
                             p_coin_pred=np.nan, exp_correct_pred=np.nan, match_acc=np.nan,
                             n_matches=len(real), picks=pk))

        for vname, feats in versions.items():
            bar.update()
            tr = df[(df.date < when) & (df.date >= pd.Timestamp(mc["train_from"]))].dropna(subset=feats)
            model = MatchModel(feats).fit(tr, mc["C"])
            k = float(mc.get("region_scale", 1.0))
            if k != 1.0:
                model.coef = [c * k if f.startswith("region_") else c for f, c in zip(feats, model.coef)]
            P = np.full((n, n), 0.5)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        P[i, j] = model.predict([[pairs[i][j][k] for k in feats]])[0]
            pm = map_from_bo3(P) if mc["series_is_bo3"] else P
            p_bo1 = stretch(pm, mc.get("bo1_stretch", 1.0))
            p_bo3 = bo3_from_map(pm)
            if st.get("all_bo3"):
                p_bo1 = p_bo3
            sim = Swiss(n, p_bo1, p_bo3, round1, sw["wins_to_advance"], sw["losses_to_eliminate"],
                        sw["bo3_for_decider"])
            w, l = sim.simulate(n_sims, seed=7)
            picks, p_coin, exp_c, _ = optimise(w, l, sizes, pc["coin_threshold"], pc["advance_counts_3_0"],
                                               restarts=4)
            pick_names = {k: [shown[i] for i in v] for k, v in picks.items()}
            correct = _score_picks(pick_names, records, pc["advance_counts_3_0"])
            # proper score: log probability the simulation gave each team's real record
            rec_lp, adv_b = [], []
            for i, t in enumerate(shown):
                a, b = map(int, records[t].split("-"))
                pr = ((w[:, i] == a) & (l[:, i] == b)).mean()
                rec_lp.append(np.log(max(pr, 1.0 / n_sims)))
                adv_b.append(((w[:, i] == 3).mean() - float(a == 3)) ** 2)
            # per-match: did the favourite win?
            hits, ll = [], []
            for wn, ln in zip(real.winner, real.loser):
                p = P[c2i[wn], c2i[ln]]
                hits.append(p > 0.5)
                ll.append(-np.log(max(p, 1e-6)))
                match_rows.append(dict(stage=st["name"], version=vname, winner=wn, loser=ln,
                                       p_winner=round(float(p), 3)))
            rows.append(dict(stage=st["name"], version=vname, correct=correct,
                             p_coin_pred=round(p_coin, 3), exp_correct_pred=round(exp_c, 2),
                             match_acc=float(np.mean(hits)) if hits else np.nan,
                             match_logloss=float(np.mean(ll)) if ll else np.nan,
                             record_logscore=float(np.mean(rec_lp)), advance_brier=float(np.mean(adv_b)),
                             n_matches=len(real), picks=pick_names))
    bar.close()
    return pd.DataFrame(rows), pd.DataFrame(match_rows)


def summarise(res: pd.DataFrame, thr: int) -> pd.DataFrame:
    g = res.groupby("version", sort=False)
    out = pd.DataFrame({
        "avg correct (of 10)": g.correct.mean().round(2),
        f"stages with {thr}+ correct": g.correct.apply(lambda c: f"{int((c >= thr).sum())}/{len(c)}"),
        "predicted P(5+)": g.p_coin_pred.mean().round(3),
        "match accuracy": g.apply(lambda x: np.average(x.match_acc, weights=x.n_matches)
                                  if x.match_acc.notna().any() else np.nan).round(3),
        "record log score (higher=better)": g.record_logscore.mean().round(4)
            if "record_logscore" in res else np.nan,
        "advance Brier (lower=better)": g.advance_brier.mean().round(4) if "advance_brier" in res else np.nan,
        "match log loss": g.apply(lambda x: np.average(x.match_logloss, weights=x.n_matches)
                                  if "match_logloss" in x and x.match_logloss.notna().any() else np.nan).round(4),
    })
    return out
