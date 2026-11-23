from pathlib import Path

import pandas as pd

from pickem import hltv
from pickem.features import FeatureBuilder, FeatureCfg

FIX = Path(__file__).parent / "fixtures"


def test_ranking_parser():
    when, df = hltv.parse_ranking_html((FIX / "ranking_snippet.html").read_text())
    assert when == pd.Timestamp("2026-11-16")
    assert df.team.tolist() == ["FURIA", "MOUZ"]
    assert df.points.tolist() == [960, 808]
    assert df.hltv_id.tolist() == [8297, 4494]


def test_results_parser():
    df = hltv.parse_results_html((FIX / "results_snippet.html").read_text())
    assert df.winner.tolist() == ["FURIA", "9z"]
    assert df.fmt.tolist() == ["bo3", "bo1"]


def test_features_use_only_past_data():
    vrs = pd.DataFrame(dict(date=pd.to_datetime(["2026-01-05", "2026-02-02"] * 2),
                            team=["A", "A", "B", "B"], rank=[1, 2, 2, 1], points=[1600, 1500, 1500, 1600]))
    m = pd.DataFrame(dict(date=pd.to_datetime(["2026-01-10", "2026-02-10"]),
                          winner=["B", "B"], loser=["A", "A"]))
    fb = FeatureBuilder(vrs, m, None, FeatureCfg())
    assert fb.vrs_points("A", "2026-02-02")[0] == 1600   # snapshot of 02-02 not yet visible
    assert fb.vrs_points("A", "2026-02-03")[0] == 1500
    assert fb.form("B", "2026-01-10")[2] == 0            # same-day match excluded
    assert fb.form("B", "2026-01-11")[0] > 0             # upset win -> positive form


def test_elo_and_h2h_use_only_past_data():
    vrs = pd.DataFrame(dict(date=pd.to_datetime(["2026-01-05"] * 2), team=["A", "B"],
                            rank=[1, 2], points=[1600, 1500]))
    m = pd.DataFrame(dict(date=pd.to_datetime(["2026-01-10", "2026-01-12"]),
                          winner=["B", "B"], loser=["A", "A"]))
    fb = FeatureBuilder(vrs, m, None, FeatureCfg(elo_k=32))
    assert fb.elo("B", "2026-01-10") == 1500            # same-day result not visible
    assert fb.elo("B", "2026-01-11") > 1500             # upset win raises Elo
    assert fb.elo("A", "2026-01-11") < 1600
    assert fb.head_to_head("A", "B", "2026-01-10") == 0.0
    assert fb.head_to_head("B", "A", "2026-01-13") > 0 > fb.head_to_head("A", "B", "2026-01-13")
