"""Feature engineering. Every feature for a match on day d uses only data
dated strictly before d (previous VRS snapshot, previous HLTV ranking, earlier
matches), so validation scores are honest.

The three requested parameters:

1. Team ranking   -> vrs_diff  (VRS points, /100) and hltv_diff (log-rank gap)
2. Opponent rating-> every feature is "team minus opponent", and sos_diff
                     measures how strong each team's recent opponents were
3. Form           -> form_diff: decay-weighted "results above expectation" over
                     the last 90 days. Beating a strong team adds a lot,
                     losing to a weak team costs a lot, so form is
                     opponent-adjusted.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

Q = np.log(10) / 400  # Valve's scale: 400 points = 10:1 odds


def elo_expect(diff_points):
    return 1.0 / (1.0 + np.exp(-Q * np.asarray(diff_points, dtype=float)))


class Names:
    """Maps announced/HLTV names to the canonical VRS name."""

    def __init__(self, aliases: dict[str, str], known: set[str]):
        self.alias = {k.lower(): v for k, v in (aliases or {}).items()}
        self.known = {k.lower(): k for k in known}

    def __call__(self, name: str) -> str:
        n = self.alias.get(name.lower(), name)
        return self.known.get(n.lower(), n)


@dataclass
class FeatureCfg:
    form_window_days: int = 90
    form_half_life_days: float = 30
    form_prior_matches: float = 3
    hltv_max_age_days: int = 45
    elo_k: float = 32
    elo_revert: float = 0.0      # at each new VRS snapshot pull Elo this share toward VRS points
    elo_new_boost: float = 0.0   # extra K for teams with few matches: K*(1+boost*exp(-n/10))
    short_form_window_days: int = 30
    short_form_half_life_days: float = 10


class FeatureBuilder:
    def __init__(self, vrs: pd.DataFrame, matches: pd.DataFrame,
                 hltv: pd.DataFrame | None, cfg: FeatureCfg,
                 regions: pd.DataFrame | None = None):
        self.cfg = cfg
        self.reg_dates, self.reg = [], {}
        if regions is not None and not regions.empty:
            self.reg_dates = sorted(regions.date.unique())
            self.reg = {d: g.set_index("team")["region"] for d, g in regions.groupby("date")}
        vrs = vrs.sort_values(["date", "rank"]).drop_duplicates(["date", "team"])
        self.vrs_dates = sorted(vrs.date.unique())
        self.vrs = {d: g.set_index("team") for d, g in vrs.groupby("date")}
        self.hltv_dates, self.hltv = [], {}
        if hltv is not None and not hltv.empty:
            hltv = hltv.sort_values(["date", "rank"]).drop_duplicates(["date", "team"])
            self.hltv_dates = sorted(hltv.date.unique())
            self.hltv = {d: g.set_index("team")["rank"] for d, g in hltv.groupby("date")}
        self._index_matches(matches)

    # -------------------------------------------------------------- rankings
    def _snap(self, dates, when):
        i = bisect.bisect_left(dates, pd.Timestamp(when)) - 1
        return dates[i] if i >= 0 else None

    def vrs_points(self, team, when) -> tuple[float, float]:
        """(points, rank) from the last snapshot before `when`. Unranked teams
        get the snapshot's lowest points and rank+1."""
        d = self._snap(self.vrs_dates, when)
        if d is None:
            return np.nan, np.nan
        t = self.vrs[d]
        if team in t.index:
            return float(t.at[team, "points"]), float(t.at[team, "rank"])
        return float(t.points.min()), float(t["rank"].max() + 1)

    def hltv_rank(self, team, when) -> float:
        d = self._snap(self.hltv_dates, when)
        if d is None or (pd.Timestamp(when) - d).days > self.cfg.hltv_max_age_days:
            return np.nan
        r = self.hltv[d]
        return float(r.get(team, len(r) + 1))

    # ------------------------------------------------------------------ form
    def _index_matches(self, matches: pd.DataFrame):
        m = matches.sort_values("date").reset_index(drop=True)
        pw = np.array([self.vrs_points(w, d)[0] for w, d in zip(m.winner, m.date)])
        pl = np.array([self.vrs_points(l, d)[0] for l, d in zip(m.loser, m.date)])
        e_w = elo_expect(pw - pl)
        hist: dict[str, list] = {}
        for i, (d, w, l) in enumerate(zip(m.date, m.winner, m.loser)):
            if np.isnan(e_w[i]):
                continue
            hist.setdefault(w, []).append((d, 1.0 - e_w[i], pl[i]))   # residual, opp points
            hist.setdefault(l, []).append((d, -e_w[i], pw[i]))
        self.hist = {}
        for t, rows in hist.items():
            ds = np.array([r[0] for r in rows], dtype="datetime64[ns]")
            self.hist[t] = (ds, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))

        # head-to-head: residuals from the alphabetically-first team's view
        h2h: dict[tuple, list] = {}
        for i, (d, w, l) in enumerate(zip(m.date, m.winner, m.loser)):
            if np.isnan(e_w[i]):
                continue
            key = tuple(sorted((w, l)))
            r = (1.0 - e_w[i]) if key[0] == w else -(1.0 - e_w[i])
            h2h.setdefault(key, []).append((d, r))
        self.h2h_hist = {k: (np.array([x[0] for x in v], dtype="datetime64[ns]"),
                             np.array([x[1] for x in v])) for k, v in h2h.items()}

        xr: dict[str, list] = {}
        if self.reg:
            for i, (d, w, l) in enumerate(zip(m.date, m.winner, m.loser)):
                if np.isnan(e_w[i]):
                    continue
                rw, rl = self.region(w, d), self.region(l, d)
                if "Unknown" in (rw, rl) or rw == rl:
                    continue
                xr.setdefault(w, []).append((d, 1.0 - e_w[i]))
                xr.setdefault(l, []).append((d, -(1.0 - e_w[i])))
        self.xreg_hist = {t: (np.array([x[0] for x in v], dtype="datetime64[ns]"),
                              np.array([x[1] for x in v])) for t, v in xr.items()}

        self._build_elo(m)

    # ---------------------------------------------------------------- region
    def region(self, team, when) -> str:
        d = self._snap(self.reg_dates, when)
        if d is None:
            return "Unknown"
        r = self.reg[d].get(team)
        if r is None:  # not ranked in that snapshot: use any earlier/later known region
            for dd in reversed(self.reg_dates):
                r = self.reg[dd].get(team)
                if r is not None:
                    break
        return r or "Unknown"

    def cross_region(self, team, when) -> float:
        """Shrunk mean result-above-expectation vs teams from OTHER regions,
        last 365 days. Positive = does better outside its region than its
        rating says (its region is underrated), negative = overrated."""
        h = self.xreg_hist.get(team)
        if h is None:
            return 0.0
        when = np.datetime64(pd.Timestamp(when))
        lo = np.searchsorted(h[0], when - np.timedelta64(365, "D"), "left")
        hi = np.searchsorted(h[0], when, "left")
        if hi <= lo:
            return 0.0
        return float(h[1][lo:hi].sum() / (hi - lo + 4))

    # ------------------------------------------------------------------- elo
    def _build_elo(self, m: pd.DataFrame):
        """Live Elo on Valve's scale, updated after every series. A team enters
        at its VRS points the first time it appears. All matches on one day use
        start-of-day ratings, so a match never sees its own result."""
        K = self.cfg.elo_k
        rating: dict[str, float] = {}
        played: dict[str, int] = {}
        post: dict[str, list] = {}
        snap_i = 0
        for d, g in m.groupby("date", sort=True):
            # new VRS snapshot(s) published since the last match day: mean-revert
            while snap_i < len(self.vrs_dates) and self.vrs_dates[snap_i] < d:
                if self.cfg.elo_revert > 0:
                    sd = self.vrs_dates[snap_i]
                    pts = self.vrs[sd]["points"]
                    floor = float(pts.min())
                    for t in rating:
                        target = float(pts[t]) if t in pts.index else floor
                        rating[t] += self.cfg.elo_revert * (target - rating[t])
                snap_i += 1
            for t in set(g.winner) | set(g.loser):
                if t not in rating:
                    p = self.vrs_points(t, d)[0]
                    rating[t] = p if not np.isnan(p) else 1000.0
            start = dict(rating)
            for w, l in zip(g.winner, g.loser):
                e = 1.0 / (1.0 + np.exp(-Q * (start[w] - start[l])))
                kw = K * (1 + self.cfg.elo_new_boost * np.exp(-played.get(w, 0) / 10))
                kl = K * (1 + self.cfg.elo_new_boost * np.exp(-played.get(l, 0) / 10))
                rating[w] += kw * (1 - e)
                rating[l] -= kl * (1 - e)
                played[w] = played.get(w, 0) + 1
                played[l] = played.get(l, 0) + 1
            for t in set(g.winner) | set(g.loser):
                post.setdefault(t, []).append((d, rating[t]))
        self.elo_hist = {t: (np.array([x[0] for x in v], dtype="datetime64[ns]"),
                             np.array([x[1] for x in v])) for t, v in post.items()}

    def elo(self, team, when) -> float:
        h = self.elo_hist.get(team)
        if h is not None:
            i = np.searchsorted(h[0], np.datetime64(pd.Timestamp(when)), "left") - 1
            if i >= 0:
                return float(h[1][i])
        p = self.vrs_points(team, when)[0]
        return p if not np.isnan(p) else 1000.0

    def head_to_head(self, a, b, when) -> float:
        """Shrunk mean of A's results above expectation vs B over the last year."""
        key = tuple(sorted((a, b)))
        h = self.h2h_hist.get(key)
        if h is None:
            return 0.0
        when = np.datetime64(pd.Timestamp(when))
        lo = np.searchsorted(h[0], when - np.timedelta64(365, "D"), "left")
        hi = np.searchsorted(h[0], when, "left")
        if hi <= lo:
            return 0.0
        v = float(h[1][lo:hi].sum() / (hi - lo + 2))
        return v if key[0] == a else -v

    def form(self, team, when, window=None, half_life=None) -> tuple[float, float, int]:
        """(form, strength_of_schedule, n_matches) over the window before `when`."""
        window = window or self.cfg.form_window_days
        half_life = half_life or self.cfg.form_half_life_days
        h = self.hist.get(team)
        if h is None:
            return 0.0, np.nan, 0
        ds, res, opp = h
        when = np.datetime64(pd.Timestamp(when))
        lo = np.searchsorted(ds, when - np.timedelta64(int(window), "D"), "left")
        hi = np.searchsorted(ds, when, "left")  # strictly before `when`
        if hi <= lo:
            return 0.0, np.nan, 0
        age = (when - ds[lo:hi]) / np.timedelta64(1, "D")
        w = 0.5 ** (age / half_life)
        f = float((w * res[lo:hi]).sum() / (w.sum() + self.cfg.form_prior_matches * 0.5))
        sos = float((w * opp[lo:hi]).sum() / w.sum())
        return f, sos, int(hi - lo)

    # -------------------------------------------------------------- features
    def team_profile(self, team, when, hltv_implied=None, hltv_weight=0.0) -> dict:
        pts, rank = self.vrs_points(team, when)
        hr = self.hltv_rank(team, when)
        if hltv_implied is not None and hltv_weight > 0 and not np.isnan(hr):
            pts = (1 - hltv_weight) * pts + hltv_weight * hltv_implied(hr)
        f, sos, n = self.form(team, when)
        fs, _, _ = self.form(team, when, self.cfg.short_form_window_days, self.cfg.short_form_half_life_days)
        return dict(team=team, vrs_points=pts, vrs_rank=rank, hltv_rank=hr,
                    form=f, sos=sos if not np.isnan(sos) else pts, n_recent=n,
                    form_short=fs, elo=self.elo(team, when), when=pd.Timestamp(when),
                    region=self.region(team, when), xreg=self.cross_region(team, when))

    def pair(self, a: dict, b: dict) -> dict:
        return dict(
            vrs_diff=(a["vrs_points"] - b["vrs_points"]) / 100.0,
            hltv_diff=np.log(b["hltv_rank"]) - np.log(a["hltv_rank"]),
            form_diff=a["form"] - b["form"],
            sos_diff=(a["sos"] - b["sos"]) / 100.0,
            elo_diff=(a["elo"] - b["elo"]) / 100.0,
            form_short_diff=a["form_short"] - b["form_short"],
            activity_diff=np.log1p(a["n_recent"]) - np.log1p(b["n_recent"]),
            h2h=self.head_to_head(a["team"], b["team"], a["when"]),
            region_americas=float(a["region"] == "Americas") - float(b["region"] == "Americas"),
            region_asia=float(a["region"] == "Asia") - float(b["region"] == "Asia"),
            xreg_diff=a["xreg"] - b["xreg"],
        )

    def training_frame(self, matches: pd.DataFrame, start, max_rank: int) -> pd.DataFrame:
        from .progress import Progress

        m = matches[matches.date >= pd.Timestamp(start)]
        rows = []
        with Progress(len(m), "Building features") as bar:
            for d, w, l in zip(m.date, m.winner, m.loser):
                bar.update()
                a, b = self.team_profile(w, d), self.team_profile(l, d)
                if np.isnan(a["vrs_points"]) or max(a["vrs_rank"], b["vrs_rank"]) > max_rank:
                    continue
                rows.append(dict(date=d, team_a=w, team_b=l, y=1, **self.pair(a, b),
                                 vrs_rank_a=a["vrs_rank"], vrs_rank_b=b["vrs_rank"],
                                 cross=a["region"] != b["region"]))
        return pd.DataFrame(rows)


def hltv_to_vrs_map(fb: FeatureBuilder, when) -> callable:
    """Fit VRS points ~ a + b*log(HLTV rank) on teams present in both current
    rankings. Used when there is no HLTV history to learn a separate weight."""
    d_h = fb._snap(fb.hltv_dates, when)
    d_v = fb._snap(fb.vrs_dates, when)
    if d_h is None or d_v is None:
        return None
    h, v = fb.hltv[d_h], fb.vrs[d_v]["points"]
    common = [t for t in h.index if t in v.index]
    x = np.log(h[common].to_numpy(float))
    y = v[common].to_numpy(float)
    b, a = np.polyfit(x, y, 1)
    return lambda rank: a + b * np.log(rank)
