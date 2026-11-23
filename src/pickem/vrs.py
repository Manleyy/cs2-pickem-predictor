"""Valve Regional Standings (VRS) data.

Valve publishes the official ranking used for Major invites in a public GitHub
repo. Every snapshot has a global standings table, and every ranked roster has a
"details" page that lists each match that fed into its score (date, opponent,
W/L). We use the tables for the ranking feature and the details pages as our
historical match dataset, so no scraping is needed for training data.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/ValveSoftware/counter-strike_regional_standings.git"
ZIP_URL = "https://codeload.github.com/ValveSoftware/counter-strike_regional_standings/zip/refs/heads/main"

_STANDING_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*\[details\]\((.+?)\)\s*\|"
)
_MATCH_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*\|\s*([WL])\s*\|"
)
_SNAPSHOT_FILE = re.compile(r"standings_global_(\d{4})_(\d{2})_(\d{2})\.md$")


def fetch_repo(dest: Path) -> Path:
    """Clone (or update) Valve's repo. Falls back to a zip download without git."""
    dest = Path(dest)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
        return dest
    if shutil.which("git"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
        return dest
    import requests

    print("git not found, downloading zip (~100 MB)...")
    r = requests.get(ZIP_URL, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        tmp = dest.parent / "_vrs_zip"
        z.extractall(tmp)
        inner = next(tmp.iterdir())
        if dest.exists():
            shutil.rmtree(dest)
        inner.rename(dest)
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def _snapshot_files(repo: Path) -> list[tuple[pd.Timestamp, Path]]:
    out = []
    for f in (Path(repo) / "live").rglob("standings_global_*.md"):
        m = _SNAPSHOT_FILE.search(f.name)
        if m:
            out.append((pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}"), f))
    return sorted(out)


def parse_standings(repo: Path) -> pd.DataFrame:
    """All global snapshots -> rows (date, rank, points, team, roster, details)."""
    rows = []
    for date, f in _snapshot_files(repo):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = _STANDING_ROW.match(line)
            if m:
                rows.append(
                    dict(date=date, rank=int(m[1]), points=int(m[2]), team=m[3].strip(),
                         roster=m[4].strip(), details=str((f.parent / m[5]).resolve()))
                )
    return pd.DataFrame(rows)


def parse_matches(repo: Path, standings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a de-duplicated series-level match list from all details pages.

    The same match shows up on both teams' pages and in many snapshots, so we key
    it by (date, team pair, winner). If a pair genuinely met twice on the same day
    with the same winner, the max count seen in any single page is kept.
    """
    if standings is None:
        standings = parse_standings(repo)
    counts: dict[tuple, int] = defaultdict(int)
    for team, path in standings[["team", "details"]].itertuples(index=False):
        p = Path(path)
        if not p.exists():
            continue
        local: dict[tuple, int] = defaultdict(int)
        for line in p.read_text(encoding="utf-8").splitlines():
            m = _MATCH_ROW.match(line)
            if not m:
                continue
            date, opp, wl = m[1], m[2].strip(), m[3]
            winner, loser = (team, opp) if wl == "W" else (opp, team)
            local[(date, winner, loser)] += 1
        for k, n in local.items():
            counts[k] = max(counts[k], n)
    rows = []
    for (date, w, l), n in counts.items():
        rows += [dict(date=pd.Timestamp(date), winner=w, loser=l, source="vrs", fmt="series")] * n
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


_REGION_FILE = re.compile(r"standings_(europe|americas|asia)_(\d{4})_(\d{2})_(\d{2})\.md$")


def parse_regions(repo: Path) -> pd.DataFrame:
    """Team -> VRS region (Europe / Americas / Asia) per snapshot."""
    rows = []
    for f in (Path(repo) / "live").rglob("standings_*.md"):
        m = _REGION_FILE.search(f.name)
        if not m:
            continue
        date = pd.Timestamp(f"{m[2]}-{m[3]}-{m[4]}")
        for line in f.read_text(encoding="utf-8").splitlines():
            r = _STANDING_ROW.match(line)
            if r:
                rows.append(dict(date=date, team=r[3].strip(), region=m[1].capitalize()))
    return pd.DataFrame(rows).drop_duplicates(["date", "team"]).sort_values("date")


def best_per_team(standings: pd.DataFrame) -> pd.DataFrame:
    """When one org has several rosters ranked (e.g. after a roster change), keep
    the best-ranked one per snapshot."""
    return standings.sort_values(["date", "rank"]).drop_duplicates(["date", "team"])
