# Third-party notices

## Leclowndu93150/pickem-prediction-model

`SIX_TEAM_PRIORITIES` in `src/pickem/swiss.py` (Valve's pairing priority table
for six-team groups) is taken from
https://github.com/Leclowndu93150/pickem-prediction-model, and the Bo1 stretch
factor of 1.6 (`model.bo1_stretch` in `config.yaml`) follows that project's
published finding. That project is licensed as follows:

```
MIT License

Copyright (c) 2026 the HLTV contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Barlow Condensed font

`src/pickem/fonts/BarlowCondensed-*.ttf` (used for `output/board.png`) are
Copyright 2017 The Barlow Project Authors (https://github.com/jpt/barlow),
licensed under the SIL Open Font License 1.1; the full license is in
`src/pickem/fonts/OFL.txt`.

## Data

- Valve's Counter-Strike Regional Standings
  (https://github.com/ValveSoftware/counter-strike_regional_standings) are
  downloaded at runtime by `python -m pickem fetch-vrs` and are not included in
  this repository.
- `data/backtest/majors.yaml` contains final Swiss-stage results taken from the
  English Wikipedia articles on the PGL/BLAST/StarLadder/IEM Majors 2024-2026,
  available under CC BY-SA 4.0.
