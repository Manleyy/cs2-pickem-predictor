# Model validation

Training matches: 19234 (from 2024-09-01, both teams VRS top 250)
Time split: fit before 2026-05-01, scored after. Coin flip log loss = 0.693.
HLTV ranking history coverage: 0% -> HLTV mode: **none (no HLTV data)** (auto keeps HLTV only if it improves held-out log loss)

## All matches

| model                                                             |   log_loss |   brier |   accuracy |    n |
|:------------------------------------------------------------------|-----------:|--------:|-----------:|-----:|
| VRS points, Valve formula (no fit)                                |     0.6792 |  0.2371 |     0.615  | 4060 |
| VRS only                                                          |     0.6559 |  0.2318 |     0.615  | 4060 |
| VRS + form                                                        |     0.6526 |  0.2303 |     0.6255 | 4060 |
| VRS + form + opponent SOS                                         |     0.6473 |  0.2279 |     0.623  | 4060 |
| Live Elo only                                                     |     0.6425 |  0.2255 |     0.6284 | 4060 |
| All 7 (incl. form + SOS)                                          |     0.64   |  0.2244 |     0.6358 | 4060 |
| CHOSEN: vrs_diff + elo_diff + h2h + region_americas + region_asia |     0.6401 |  0.2244 |     0.6346 | 4060 |

## Both teams in VRS top 60 (Major level)

| model                                                             |   log_loss |   brier |   accuracy |   n |
|:------------------------------------------------------------------|-----------:|--------:|-----------:|----:|
| VRS points, Valve formula (no fit)                                |     0.7396 |  0.2596 |     0.5749 | 994 |
| VRS only                                                          |     0.6717 |  0.2394 |     0.5749 | 994 |
| VRS + form                                                        |     0.6705 |  0.2388 |     0.5815 | 994 |
| VRS + form + opponent SOS                                         |     0.6686 |  0.2379 |     0.5885 | 994 |
| Live Elo only                                                     |     0.6639 |  0.2355 |     0.5976 | 994 |
| All 7 (incl. form + SOS)                                          |     0.6627 |  0.2349 |     0.5905 | 994 |
| CHOSEN: vrs_diff + elo_diff + h2h + region_americas + region_asia |     0.663  |  0.235  |     0.5986 | 994 |

## Final model

```
feature        coef/std  (per 1 unit of raw feature)
  vrs_diff     +0.058    +0.021
  elo_diff     +0.644    +0.269
  h2h          +0.087    +0.631
  region_americas -0.068    -0.247
  region_asia  -0.047    -0.233
```

## Calibration (final model, all data)

| bin        |   predicted |   actual |    n |
|:-----------|------------:|---------:|-----:|
| (0.0, 0.1] |       0.078 |    0.181 |  116 |
| (0.1, 0.2] |       0.164 |    0.205 |  969 |
| (0.2, 0.3] |       0.258 |    0.244 | 2896 |
| (0.3, 0.4] |       0.355 |    0.345 | 6069 |
| (0.4, 0.5] |       0.452 |    0.44  | 9223 |
| (0.5, 0.6] |       0.548 |    0.56  | 9145 |
| (0.6, 0.7] |       0.645 |    0.655 | 6069 |
| (0.7, 0.8] |       0.742 |    0.756 | 2896 |
| (0.8, 0.9] |       0.836 |    0.795 |  969 |
| (0.9, 1.0] |       0.922 |    0.819 |  116 |