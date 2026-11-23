# Backtest on past Majors

11 Swiss stages, each model refit on data from before the stage only. 20,000 simulations per stage. Pick'Em scored as 2 x 3-0, 6 x advance (3-1/3-2), 2 x 0-3.

## Summary

| version                                                            |   avg correct (of 10) | stages with 5+ correct   |   predicted P(5+) |   match accuracy |   record log score (higher=better) |   advance Brier (lower=better) |   match log loss |
|:-------------------------------------------------------------------|----------------------:|:-------------------------|------------------:|-----------------:|-----------------------------------:|-------------------------------:|-----------------:|
| Pick by seed (listing order)                                       |                  3.55 | 3/11                     |           nan     |          nan     |                           nan      |                       nan      |         nan      |
| Pick by VRS points                                                 |                  4.27 | 5/11                     |           nan     |          nan     |                           nan      |                       nan      |         nan      |
| VRS only                                                           |                  4.09 | 6/11                     |             0.289 |            0.623 |                            -1.6666 |                         0.2081 |           0.651  |
| Original (VRS+form+SOS)                                            |                  3.64 | 3/11                     |             0.39  |            0.626 |                            -1.6942 |                         0.2147 |           0.6598 |
| Form model (VRS+form+SOS+Elo+h2h+region)                           |                  4    | 6/11                     |             0.418 |            0.631 |                            -1.6635 |                         0.2074 |           0.6468 |
| Current: vrs_diff + elo_diff + h2h + region_americas + region_asia |                  4.45 | 7/11                     |             0.401 |            0.64  |                            -1.6461 |                         0.2037 |           0.6405 |

## Correct picks per stage

| stage                           |   Pick by seed (listing order) |   Pick by VRS points |   VRS only |   Original (VRS+form+SOS) |   Form model (VRS+form+SOS+Elo+h2h+region) |   Current: vrs_diff + elo_diff + h2h + region_americas + region_asia |
|:--------------------------------|-------------------------------:|---------------------:|-----------:|--------------------------:|-------------------------------------------:|---------------------------------------------------------------------:|
| Shanghai 2024 Opening Stage     |                              3 |                    3 |          3 |                         3 |                                          5 |                                                                    7 |
| Shanghai 2024 Elimination Stage |                              5 |                    5 |          6 |                         4 |                                          4 |                                                                    6 |
| Austin 2025 Stage 1             |                              6 |                    6 |          6 |                         5 |                                          5 |                                                                    4 |
| Austin 2025 Stage 2             |                              3 |                    2 |          2 |                         4 |                                          2 |                                                                    2 |
| Austin 2025 Stage 3             |                              3 |                    4 |          6 |                         2 |                                          3 |                                                                    5 |
| Budapest 2025 Stage 1           |                              5 |                    6 |          5 |                         3 |                                          3 |                                                                    3 |
| Budapest 2025 Stage 2           |                              3 |                    5 |          6 |                         4 |                                          5 |                                                                    5 |
| Budapest 2025 Stage 3           |                              1 |                    4 |          4 |                         5 |                                          5 |                                                                    5 |
| Cologne 2026 Stage 1            |                              4 |                    4 |          1 |                         5 |                                          5 |                                                                    5 |
| Cologne 2026 Stage 2            |                              3 |                    5 |          5 |                         4 |                                          5 |                                                                    5 |
| Cologne 2026 Stage 3            |                              3 |                    3 |          1 |                         1 |                                          2 |                                                                    2 |

## Current model's picks

- **Shanghai 2024 Opening Stage** (7/10): 3-0 The MongolZ, Liquid | advance FURIA, Virtus.pro, BIG, Complexity, GamerLegion, paiN | 0-3 Rare Atom, Imperial
- **Shanghai 2024 Elimination Stage** (6/10): 3-0 Vitality, Natus Vincere | advance G2, Spirit, MOUZ, FaZe, The MongolZ, Liquid | 0-3 paiN, MIBR
- **Austin 2025 Stage 1** (4/10): 3-0 HEROIC, Complexity | advance B8, BetBoom, OG, FlyQuest, Imperial, Wildcard | 0-3 Chinggis Warriors, Metizport
- **Austin 2025 Stage 2** (2/10): 3-0 Falcons, 3DMAX | advance FaZe, HEROIC, Virtus.pro, B8, MIBR, FURIA | 0-3 Legacy, Lynn Vision
- **Austin 2025 Stage 3** (5/10): 3-0 Vitality, Spirit | advance The MongolZ, Natus Vincere, MOUZ, G2, 3DMAX, FaZe | 0-3 Legacy, paiN
- **Budapest 2025 Stage 1** (3/10): 3-0 Legacy, PARIVISION | advance Lynn Vision, FaZe, B8, GamerLegion, NRG, fnatic | 0-3 RED Canids, The Huns
- **Budapest 2025 Stage 2** (5/10): 3-0 Natus Vincere, Aurora | advance Astralis, FaZe, Liquid, B8, 3DMAX, PARIVISION | 0-3 Passion UA, Imperial
- **Budapest 2025 Stage 3** (5/10): 3-0 Vitality, FURIA | advance Falcons, MOUZ, Spirit, Natus Vincere, The MongolZ, G2 | 0-3 Imperial, Passion UA
- **Cologne 2026 Stage 1** (5/10): 3-0 GamerLegion, BetBoom | advance B8, MIBR, Lynn Vision, BIG, M80, HEROIC | 0-3 Gaimin Gladiators, THUNDER dOWNUNDER
- **Cologne 2026 Stage 2** (5/10): 3-0 Spirit, G2 | advance FUT, Astralis, Legacy, BetBoom, GamerLegion, B8 | 0-3 FlyQuest, TYLOO
- **Cologne 2026 Stage 3** (2/10): 3-0 Vitality, Falcons | advance Natus Vincere, Spirit, MOUZ, FUT, BetBoom, Aurora | 0-3 9z, Monte