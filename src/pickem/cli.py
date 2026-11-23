"""Command line entry point:  python -m pickem <command>

  fetch-vrs     download Valve's VRS repo and build processed CSVs
  build-vrs     re-parse an already downloaded VRS repo
  hltv-fetch    download HLTV ranking history / recent results (polite, cached)
  hltv-parse    parse saved HLTV pages (auto or Ctrl+S) into processed CSVs
  train         build features, validate on a time split, fit and save the model
  predict       simulate the Swiss stage and choose Pick'Em picks
  all           train + predict
  backtest      replay past Majors: how many Pick'Em picks would have been right
  board         redraw output/board.png from output/predictions.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import hltv, vrs
from .features import FeatureBuilder, FeatureCfg, Names, hltv_to_vrs_map
from .model import MatchModel, bo3_from_map, calibration_table, evaluate, map_from_bo3, stretch
from .optimize import optimise
from .swiss import Swiss, record_probs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROC = DATA / "processed"
RAW = DATA / "raw"
OUT = ROOT / "output"
MODEL_PATH = ROOT / "models" / "model.json"


def apply_region_scale(model, cfg):
    """Multiply the learned region penalty by model.region_scale (1.0 = as learned)."""
    k = float(cfg["model"].get("region_scale", 1.0))
    if k != 1.0:
        model.coef = [c * k if f.startswith("region_") else c for f, c in zip(model.features, model.coef)]
    return model


def load_cfg(path=None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------- data ---
def cmd_build_vrs(repo: Path):
    PROC.mkdir(parents=True, exist_ok=True)
    s = vrs.parse_standings(repo)
    m = vrs.parse_matches(repo, s)
    vrs.parse_regions(repo).to_csv(PROC / "vrs_regions.csv", index=False)
    s.drop(columns="details").to_csv(PROC / "vrs_standings.csv", index=False)
    m.to_csv(PROC / "vrs_matches.csv", index=False)
    print(f"VRS: {s.date.nunique()} snapshots (latest {s.date.max().date()}), "
          f"{len(m)} series {m.date.min().date()} .. {m.date.max().date()}")


def load_data(cfg):
    if not (PROC / "vrs_standings.csv").exists():
        raise SystemExit(
            "No VRS data found in data/processed/.\n"
            "Download Valve's ranking data first (once, about 150 MB):\n\n"
            "    python -m pickem fetch-vrs\n")
    s = pd.read_csv(PROC / "vrs_standings.csv", parse_dates=["date"])
    m = pd.read_csv(PROC / "vrs_matches.csv", parse_dates=["date"])
    # Valve sometimes changes a team's capitalisation (BetBoom -> BETBOOM).
    # Map every spelling to the newest one so each team has one history.
    newest = {}
    for t in pd.concat([m[["date", "winner"]].rename(columns={"winner": "team"}),
                        m[["date", "loser"]].rename(columns={"loser": "team"}),
                        s[["date", "team"]]]).sort_values("date", kind="stable").team:
        newest[t.lower()] = t
    canon = lambda x: newest.get(x.lower(), x)
    s["team"] = s.team.map(canon)
    m["winner"] = m.winner.map(canon)
    m["loser"] = m.loser.map(canon)
    names = Names(cfg.get("aliases"), set(newest.values()))
    last_vrs = m.date.max()
    extra = []
    if (PROC / "hltv_results.csv").exists():
        extra.append(pd.read_csv(PROC / "hltv_results.csv", parse_dates=["date"]))
    manual = DATA / "manual" / "extra_matches.csv"
    if manual.exists():
        mm = pd.read_csv(manual, parse_dates=["date"], comment="#")
        if not mm.empty:
            extra.append(mm.assign(source="manual"))
    for e in extra:  # only matches after the VRS data ends, to avoid duplicates
        e = e[e.date > last_vrs].copy()
        e["winner"] = e.winner.map(names)
        e["loser"] = e.loser.map(names)
        m = pd.concat([m, e[["date", "winner", "loser", "source", "fmt"]]], ignore_index=True)
    m = m.sort_values("date").reset_index(drop=True)
    h = None
    if (PROC / "hltv_rankings.csv").exists():
        h = pd.read_csv(PROC / "hltv_rankings.csv", parse_dates=["date"])
        h["team"] = h.team.map(names)
    reg = None
    if (PROC / "vrs_regions.csv").exists():
        reg = pd.read_csv(PROC / "vrs_regions.csv", parse_dates=["date"])
        reg["team"] = reg.team.map(canon)
        reg = reg.drop_duplicates(["date", "team"])
    return s, m, h, names, reg


def builder(cfg, s, m, h, reg=None) -> FeatureBuilder:
    keys = FeatureCfg.__dataclass_fields__
    return FeatureBuilder(s, m, h, FeatureCfg(**{k: v for k, v in cfg["features"].items() if k in keys}), reg)


# ------------------------------------------------------------------ train ---
def cmd_train(cfg):
    s, m, h, _, reg = load_data(cfg)
    fb = builder(cfg, s, m, h, reg)
    mc = cfg["model"]
    df = fb.training_frame(m, mc["train_from"], mc["max_vrs_rank"])
    hltv_cov = df.hltv_diff.notna().mean()
    old = ["vrs_diff", "form_diff", "sos_diff"]
    base = list(mc.get("features") or ["vrs_diff", "elo_diff", "h2h", "region_americas", "region_asia"])
    sets = {"VRS only": ["vrs_diff"], "VRS + form": ["vrs_diff", "form_diff"],
            "VRS + form + opponent SOS": old,
            "Live Elo only": ["elo_diff"],
            "All 7 (incl. form + SOS)": ["vrs_diff", "form_diff", "sos_diff", "elo_diff", "h2h",
                                         "region_americas", "region_asia"],
            "CHOSEN: " + " + ".join(base): base}
    with_hltv = base + ["hltv_diff"]
    if hltv_cov > 0.05:
        sets["HLTV only"] = ["hltv_diff"]
        sets["CHOSEN + HLTV"] = with_hltv

    ev = evaluate(df, mc["test_from"], sets, mc["C"])
    top = df[(df.vrs_rank_a <= 60) & (df.vrs_rank_b <= 60)]
    ev_top = evaluate(top, mc["test_from"], sets, mc["C"])

    # HLTV: "auto" keeps it only if it lowers held-out log loss (all matches)
    choice = str(mc.get("use_hltv", "auto")).lower()
    ll = ev.set_index("model")["log_loss"]
    if not fb.hltv_dates:
        hltv_mode = "none (no HLTV data)"
    elif hltv_cov < 0.3:
        hltv_mode = "consensus"          # too little history to learn a weight
    elif choice in ("yes", "true"):
        hltv_mode = "learned"
    elif choice in ("no", "false"):
        hltv_mode = "off"
    else:
        better = ll.get("CHOSEN + HLTV", 9) < ll.get("CHOSEN: " + " + ".join(base), 9)
        hltv_mode = "learned" if better else "off"
    feats = with_hltv if hltv_mode == "learned" else base
    fit_df = df.dropna(subset=feats)
    model = MatchModel(feats).fit(fit_df, mc["C"])
    model.meta = dict(trained=str(date.today()), n_matches=len(fit_df),
                      hltv_mode=hltv_mode,
                      hltv_coverage=round(float(hltv_cov), 3),
                      data_until=str(m.date.max().date()))
    model.save(MODEL_PATH)

    report = ["# Model validation", "",
              f"Training matches: {len(fit_df)} (from {mc['train_from']}, both teams VRS top {mc['max_vrs_rank']})",
              f"Time split: fit before {mc['test_from']}, scored after. Coin flip log loss = 0.693.",
              f"HLTV ranking history coverage: {hltv_cov:.0%} -> HLTV mode: **{hltv_mode}** (auto keeps HLTV only if it improves held-out log loss)", "",
              "## All matches", "", ev.to_markdown(index=False), "",
              "## Both teams in VRS top 60 (Major level)", "", ev_top.to_markdown(index=False), "",
              "## Final model", "", "```", model.describe(), "```", "",
              "## Calibration (final model, all data)", "",
              calibration_table(fit_df, model).to_markdown()]
    OUT.mkdir(exist_ok=True)
    (OUT / "validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:8]))
    print(ev.to_string(index=False)); print("\nTop-60 only:"); print(ev_top.to_string(index=False))
    print("\n" + model.describe())
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and output/validation.md")


# ---------------------------------------------------------------- predict ---
def cmd_predict(cfg, n_sims=None):
    s, m, h, names, reg = load_data(cfg)
    fb = builder(cfg, s, m, h, reg)
    model = MatchModel.load(MODEL_PATH)
    apply_region_scale(model, cfg)
    t = cfg["tournament"]
    when = pd.Timestamp(t["date"])
    shown = t["seeds"]
    canon = [names(x) for x in shown]
    idx = {x: i for i, x in enumerate(shown)}

    implied, wgt = None, 0.0
    if model.meta.get("hltv_mode") == "consensus" and fb.hltv_dates:
        implied = hltv_to_vrs_map(fb, when)
        wgt = cfg["features"]["hltv_weight_if_no_history"] if implied else 0.0
    prof = [fb.team_profile(c, when, implied, wgt) for c in canon]
    for p, c in zip(prof, canon):
        p["vrs_points_raw"] = fb.vrs_points(c, when)[0]

    n = len(shown)
    p_series = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i != j:
                f = fb.pair(prof[i], prof[j])
                p_series[i, j] = model.predict([[f[k] for k in model.features]])[0]
    p_map = map_from_bo3(p_series) if cfg["model"]["series_is_bo3"] else p_series
    # Bo1 favourites win more often than an average map suggests (the veto
    # leaves the underdog no comfort pick), so Bo1 log-odds are stretched.
    p_bo1, p_bo3 = stretch(p_map, cfg["model"].get("bo1_stretch", 1.0)), bo3_from_map(p_map)

    r1 = t.get("round1") or [[shown[i], shown[i + n // 2]] for i in range(n // 2)]
    round1 = [(idx[a], idx[b]) for a, b in r1]
    played = [(idx[a], idx[b]) for a, b in (t.get("played") or [])]
    sw = cfg["swiss"]
    sim = Swiss(n, p_bo1, p_bo3, round1, sw["wins_to_advance"], sw["losses_to_eliminate"],
                sw["bo3_for_decider"], played)
    N = n_sims or sw["n_sims"]
    wins, losses = sim.simulate(N, seed=42, show_progress=True)
    rec = record_probs(wins, losses, shown)

    pc = cfg["pickem"]
    sizes = {"3-0": pc["n_3_0"], "0-3": pc["n_0_3"], "advance": pc["n_advance"]}
    locked = {k: [idx[x] for x in v] for k, v in (pc.get("locked") or {}).items()}
    picks, p_coin, exp_c, dist = optimise(wins, losses, sizes, pc["coin_threshold"],
                                          pc["advance_counts_3_0"], locked, show_progress=True)
    pick_names = {k: [shown[i] for i in v] for k, v in picks.items()}

    # ---- outputs
    OUT.mkdir(exist_ok=True)
    prof_df = pd.DataFrame(prof).assign(seed=range(1, n + 1), name=shown)[
        ["seed", "name", "team", "region", "vrs_points_raw", "vrs_rank", "hltv_rank", "vrs_points",
         "elo", "form", "sos", "n_recent"]]
    prof_df.to_csv(OUT / "team_profiles.csv", index=False)
    rec.to_csv(OUT / "record_probabilities.csv", index=False)
    pd.DataFrame(p_bo1, index=shown, columns=shown).round(3).to_csv(OUT / "matrix_bo1.csv")
    pd.DataFrame(p_bo3, index=shown, columns=shown).round(3).to_csv(OUT / "matrix_bo3.csv")
    r1_rows = [dict(team_a=shown[a], team_b=shown[b], p_a_bo1=round(float(p_bo1[a, b]), 3))
               for a, b in round1]
    result = dict(tournament=t["name"], generated=pd.Timestamp.now().isoformat(timespec="minutes"),
                  n_sims=N, model=model.meta, picks=pick_names,
                  p_coin=round(p_coin, 4), expected_correct=round(exp_c, 3),
                  correct_distribution=[round(float(x), 4) for x in dist],
                  round1=r1_rows, records=rec.round(4).to_dict("records"),
                  profiles=prof_df.round(4).astype(object).where(prof_df.notna(), None).to_dict("records"))
    (OUT / "predictions.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    md = [f"# {t['name']} - Pick'Em", "",
          f"{N:,} simulations · model data until {model.meta.get('data_until')} · HLTV mode: {model.meta.get('hltv_mode')}", "",
          "## Picks", ""]
    for slot in ("3-0", "advance", "0-3"):
        for x in pick_names[slot]:
            pr = rec.set_index("team").loc[x]
            p = pr["3-0"] if slot == "3-0" else pr["0-3"] if slot == "0-3" else pr["3-1"] + pr["3-2"]
            md.append(f"- **{slot}**: {x} ({p:.0%})")
    md += ["", f"P(at least {pc['coin_threshold']} correct) = **{p_coin:.1%}**, expected correct = {exp_c:.2f} / 10", "",
           "## Round 1 (Bo1)", "", pd.DataFrame(r1_rows).to_markdown(index=False), "",
           "## Record probabilities", "", (rec.set_index("team") * 100).round(1).to_markdown(), "",
           "## Team inputs", "", prof_df.round(3).to_markdown(index=False)]
    (OUT / "picks.md").write_text("\n".join(md), encoding="utf-8")
    try:
        from .board import draw_board
        draw_board(OUT / "predictions.json", OUT / "board.png")
        print("Wrote output/board.png")
    except ImportError:
        print("(install pillow to also draw output/board.png)")

    pd.set_option("display.width", 160)
    print(prof_df.round(3).to_string(index=False))
    print("\nRound 1 (Bo1):"); print(pd.DataFrame(r1_rows).to_string(index=False))
    print("\nRecord probabilities (%):"); print((rec.set_index("team") * 100).round(1).to_string())
    print("\nPICKS")
    for slot in ("3-0", "advance", "0-3"):
        print(f"  {slot:<8}", ", ".join(pick_names[slot]))
    print(f"\nP(>= {pc['coin_threshold']} correct) = {p_coin:.1%}   expected correct = {exp_c:.2f}")
    print("Wrote output/picks.md, output/predictions.json and CSVs")


def cmd_backtest(cfg, n_sims=None):
    from . import backtest

    s, m, h, names, reg = load_data(cfg)
    fb = builder(cfg, s, m, h, reg)
    mc = cfg["model"]
    df = fb.training_frame(m, mc["train_from"], mc["max_vrs_rank"])
    current = list(mc.get("features"))
    versions = {"VRS only": ["vrs_diff"],
                "Original (VRS+form+SOS)": ["vrs_diff", "form_diff", "sos_diff"],
                "Form model (VRS+form+SOS+Elo+h2h+region)": ["vrs_diff", "form_diff", "sos_diff", "elo_diff",
                                                            "h2h", "region_americas", "region_asia"],
                "Current: " + " + ".join(current): current}
    res, per_match = backtest.run(cfg, fb, df, m, names, DATA / "backtest" / "majors.yaml",
                                  n_sims or 20000, versions)
    thr = cfg["pickem"]["coin_threshold"]
    summ = backtest.summarise(res, thr)
    OUT.mkdir(exist_ok=True)
    table = res.pivot(index="stage", columns="version", values="correct")
    table = table.loc[res.stage.unique(), res.version.unique()]
    res.to_csv(OUT / "backtest_stages.csv", index=False)
    per_match.to_csv(OUT / "backtest_matches.csv", index=False)
    cur = res[res.version.str.startswith("Current")]
    lines = ["# Backtest on past Majors", "",
             f"{res.stage.nunique()} Swiss stages, each model refit on data from before the stage only. "
             f"{n_sims or 20000:,} simulations per stage. Pick'Em scored as 2 x 3-0, 6 x advance (3-1/3-2), 2 x 0-3.", "",
             "## Summary", "", summ.to_markdown(), "",
             "## Correct picks per stage", "", table.to_markdown(), "",
             "## Current model's picks", ""]
    for _, r in cur.iterrows():
        p = r.picks
        lines.append(f"- **{r.stage}** ({r.correct}/10): 3-0 {', '.join(p['3-0'])} | "
                     f"advance {', '.join(p['advance'])} | 0-3 {', '.join(p['0-3'])}")
    (OUT / "backtest.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.to_string()); print(); print(table.to_string())
    print("\nWrote output/backtest.md, backtest_stages.csv, backtest_matches.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pickem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch-vrs", "build-vrs", "hltv-fetch", "hltv-parse",
                                        "train", "predict", "all", "backtest", "board"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--sims", type=int, default=None, help="override number of simulations")
    ap.add_argument("--rankings-from", default="2024-09-02", help="hltv-fetch: first ranking date")
    ap.add_argument("--rankings-step", type=int, default=4, help="hltv-fetch: weeks between rankings")
    ap.add_argument("--results-from", default=None, help="hltv-fetch: also fetch results from this date")
    ap.add_argument("--skip-rankings", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    repo = RAW / "vrs"

    if a.command == "fetch-vrs":
        vrs.fetch_repo(repo); cmd_build_vrs(repo)
    elif a.command == "build-vrs":
        cmd_build_vrs(repo)
    elif a.command == "hltv-fetch":
        end = pd.Timestamp(cfg["tournament"]["date"]).date()
        try:
            if not a.skip_rankings:
                hltv.fetch_rankings(date.fromisoformat(a.rankings_from), end,
                                    RAW / "hltv" / "rankings", a.rankings_step)
            if a.results_from:
                hltv.fetch_results(date.fromisoformat(a.results_from), end, RAW / "hltv" / "results")
        except hltv.BlockedError as e:
            print(e)
        hltv.build_tables(RAW / "hltv", PROC)
    elif a.command == "hltv-parse":
        hltv.build_tables(RAW / "hltv", PROC)
    elif a.command == "train":
        cmd_train(cfg)
    elif a.command == "predict":
        cmd_predict(cfg, a.sims)
    elif a.command == "board":
        from .board import draw_board
        draw_board(OUT / "predictions.json", OUT / "board.png")
        print("Wrote output/board.png")
    elif a.command == "backtest":
        cmd_backtest(cfg, a.sims)
    elif a.command == "all":
        cmd_train(cfg); print("\n" + "=" * 60 + "\n"); cmd_predict(cfg, a.sims)


if __name__ == "__main__":
    main()
