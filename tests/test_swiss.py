import numpy as np

from pickem.model import bo3_from_map, map_from_bo3
from pickem.optimize import optimise
from pickem.swiss import Swiss, _pair_group

N = 16
R1 = [(i, i + 8) for i in range(8)]


def make(p=0.5):
    P = np.full((N, N), p)
    return Swiss(N, P, bo3_from_map(P), R1)


def test_swiss_invariants():
    sw = make()
    rng = np.random.default_rng(1)
    for _ in range(300):
        w, l = sw.run_once(rng)
        assert ((w == 3) | (l == 3)).all()
        assert (w == 3).sum() == 8
        assert ((w == 3) & (l == 0)).sum() == 2
        assert ((w == 0) & (l == 3)).sum() == 2
        assert w.sum() == l.sum() == 33  # 8+8+8+6+3 matches


def test_no_rematches():
    sw = make()
    rng = np.random.default_rng(2)
    seen_pairs = []
    orig = sw.pairings

    def spy(w, l, opps, *a):
        out = orig(w, l, opps, *a)
        for a, b in out:
            assert b not in opps[a]
        return out

    sw.pairings = spy
    for _ in range(200):
        sw.run_once(rng)


def test_pair_group_avoids_seen():
    seen = {frozenset((0, 3))}
    assert _pair_group([0, 1, 2, 3], seen) == [(0, 2), (1, 3)]


def test_strong_team_advances_more():
    P = np.full((N, N), 0.5)
    P[0, :] = 0.8; P[:, 0] = 0.2; P[0, 0] = 0.5
    sw = Swiss(N, P, bo3_from_map(P), R1)
    w, l = sw.simulate(2000, seed=0)
    assert (w[:, 0] == 3).mean() > 0.85


def test_played_results_are_respected():
    P = np.full((N, N), 0.5)
    sw = Swiss(N, P, P, R1, played=[(8, 0)])  # seed 9 beats seed 1 in round 1
    w, l = sw.simulate(200, seed=0)
    assert (l[:, 0] >= 1).all()


def test_bo3_inverse():
    q = np.linspace(0.01, 0.99, 50)
    assert np.allclose(bo3_from_map(map_from_bo3(q)), q, atol=1e-6)


def test_optimiser_valid_picks():
    w, l = make().simulate(1000, seed=3)
    picks, p, e, dist = optimise(w, l, {"3-0": 2, "0-3": 2, "advance": 6})
    flat = [t for v in picks.values() for t in v]
    assert len(flat) == len(set(flat)) == 10
    assert 0 <= p <= 1 and abs(dist.sum() - 1) < 1e-9


def test_six_team_priority_table_used_in_round_4():
    # six 2-1 teams in round 4 (sorted order 0..5): no rematches -> 1st pattern 1v6, 2v5, 3v4
    from pickem.swiss import SIX_TEAM_PRIORITIES
    assert SIX_TEAM_PRIORITIES[0] == ((0, 5), (1, 4), (2, 3))
    assert SIX_TEAM_PRIORITIES[2] == ((0, 4), (1, 5), (2, 3))
