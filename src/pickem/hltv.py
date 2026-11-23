"""HLTV data: world ranking (weekly, Mondays) and match results.

HLTV sits behind Cloudflare and has no public API, so there are two ways in:

1. Automatic (`pickem hltv-fetch`): polite requests with `curl_cffi` (browser
   TLS fingerprint), a 4-8 s random delay and an on-disk cache so each page is
   fetched once. Works most of the time from a home connection. If HLTV answers
   403 / "Just a moment...", stop; don't hammer it.
2. Manual fallback: open the page in your browser, Ctrl+S ("Webpage, HTML
   only") into data/raw/hltv/rankings/ or data/raw/hltv/results/, then run
   `pickem hltv-parse`. The parser reads exactly the same HTML.

Keep volume low and the data for personal use; HLTV doesn't offer it as an API.
"""
from __future__ import annotations

import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.hltv.org"
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


class BlockedError(RuntimeError):
    pass


# ----------------------------------------------------------------- parsing ---
def parse_ranking_html(html: str) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """Parse an HLTV world ranking page -> (ranking date, DataFrame)."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for el in soup.select(".ranked-team"):
        pos = el.select_one(".position")
        name = el.select_one(".name")
        pts = el.select_one(".points")
        link = el.select_one("a.moreLink")
        if not (pos and name):
            continue
        team_id = None
        if link and (m := re.search(r"/team/(\d+)/", link.get("href", ""))):
            team_id = int(m[1])
        rows.append(dict(
            rank=int(re.sub(r"\D", "", pos.get_text())),
            team=name.get_text(strip=True),
            points=int(re.sub(r"\D", "", pts.get_text())) if pts else None,
            hltv_id=team_id,
        ))
    when = None
    head = soup.select_one(".regional-ranking-header")
    if head and (m := re.search(r"on (\w+) (\d+)\w*, (\d{4})", head.get_text())):
        when = pd.Timestamp(f"{m[1]} {m[2]} {m[3]}")
    return when, pd.DataFrame(rows)


def parse_results_html(html: str) -> pd.DataFrame:
    """Parse an HLTV /results page -> one row per series.

    `fmt` is 'bo1' when HLTV shows a map name, else 'bo3'/'bo5' as shown.
    Forfeits ('def') are dropped.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for sub in soup.select(".results-all .results-sublist"):
        head = sub.select_one(".standard-headline")
        day = None
        if head and (m := re.search(r"(\w+) (\d+)\w*,? (\d{4})", head.get_text())):
            day = pd.Timestamp(f"{m[1]} {m[2]} {m[3]}")
        for con in sub.select(".result-con"):
            t1 = con.select_one(".team1 .team")
            t2 = con.select_one(".team2 .team")
            score = con.select(".result-score span")
            if not (t1 and t2 and len(score) == 2):
                continue
            s1, s2 = (int(s.get_text(strip=True)) for s in score)
            fmt_raw = (con.select_one(".map-text") or con.select_one(".map")).get_text(strip=True).lower() \
                if (con.select_one(".map-text") or con.select_one(".map")) else ""
            if fmt_raw == "def":
                continue
            fmt = fmt_raw if fmt_raw in ("bo3", "bo5", "bo2") else "bo1"
            ev = con.select_one(".event-name")
            n1, n2 = t1.get_text(strip=True), t2.get_text(strip=True)
            if s1 == s2:
                continue  # draws (bo2) carry no winner
            w, l = (n1, n2) if s1 > s2 else (n2, n1)
            rows.append(dict(date=day, winner=w, loser=l, score=f"{max(s1,s2)}-{min(s1,s2)}",
                             fmt=fmt, event=ev.get_text(strip=True) if ev else None, source="hltv"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fetching ---
def _session():
    try:
        from curl_cffi import requests as cr
        return cr.Session(impersonate="chrome")
    except ImportError:  # plain requests is very likely to be blocked
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
        return s


def fetch_cached(url: str, cache_file: Path, session=None, min_delay=4.0, max_delay=8.0) -> str:
    cache_file = Path(cache_file)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    session = session or _session()
    time.sleep(random.uniform(min_delay, max_delay))
    r = session.get(url, timeout=30)
    text = r.text
    if r.status_code in (403, 429, 503) or "Just a moment" in text[:3000]:
        raise BlockedError(
            f"HLTV blocked the request ({r.status_code}) for {url}.\n"
            f"Open it in your browser, save it with Ctrl+S as\n  {cache_file}\n"
            "and run the command again (cached files are reused).")
    r.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def ranking_url(d: date) -> str:
    return f"{BASE}/ranking/teams/{d.year}/{MONTHS[d.month - 1]}/{d.day}"


def mondays(start: date, end: date, step_weeks: int = 1):
    d = start + timedelta(days=(7 - start.weekday()) % 7)
    while d <= end:
        yield d
        d += timedelta(weeks=step_weeks)


def fetch_rankings(start: date, end: date, raw_dir: Path, step_weeks: int = 4) -> None:
    """Download ranking pages (every `step_weeks` Mondays) into raw_dir."""
    s = _session()
    for d in mondays(start, end, step_weeks):
        fetch_cached(ranking_url(d), Path(raw_dir) / f"{d.isoformat()}.html", s)
        print("ranking", d)


def fetch_results(start: date, end: date, raw_dir: Path, max_pages: int = 20) -> None:
    """Download /results pages (100 series each) for a date window."""
    s = _session()
    for page in range(max_pages):
        url = f"{BASE}/results?startDate={start}&endDate={end}&offset={page * 100}"
        html = fetch_cached(url, Path(raw_dir) / f"{start}_{end}_{page:02d}.html", s)
        n = len(parse_results_html(html))
        print("results page", page, n, "series")
        if n < 100:
            break


def build_tables(raw_dir: Path, out_dir: Path) -> None:
    """Parse every saved HLTV page into processed CSVs."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = []
    for f in sorted((raw_dir / "rankings").glob("*.htm*")):
        when, df = parse_ranking_html(f.read_text(encoding="utf-8", errors="ignore"))
        if df.empty:
            print("skip (no ranking found):", f.name)
            continue
        if when is None:
            when = pd.Timestamp(re.search(r"\d{4}-\d{2}-\d{2}", f.name)[0])
        ranks.append(df.assign(date=when))
    for f in sorted((raw_dir / "rankings").glob("*.csv")):
        ranks.append(pd.read_csv(f, parse_dates=["date"]))
    if ranks:
        r = pd.concat(ranks).drop_duplicates(["date", "team"]).sort_values(["date", "rank"])
        r.to_csv(out_dir / "hltv_rankings.csv", index=False)
        print("hltv_rankings.csv:", r.date.nunique(), "rankings")
    res = [parse_results_html(f.read_text(encoding="utf-8", errors="ignore"))
           for f in sorted((raw_dir / "results").glob("*.htm*"))]
    res = [r for r in res if not r.empty]
    if res:
        r = pd.concat(res).drop_duplicates().sort_values("date")
        r.to_csv(out_dir / "hltv_results.csv", index=False)
        print("hltv_results.csv:", len(r), "series")
