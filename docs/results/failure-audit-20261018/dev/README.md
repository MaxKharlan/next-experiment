# Saved benchmark failure inventory

Generated from saved score rows. Categories overlap; their counts are not additive.
Missing fix measurements are reported separately, not counted as failed fixes.

| Method | Cases | Cases with findings | Wrong confirmations | Confirmed without measured fix |
|---|---:|---:|---:|---:|
| single_ultra | 5 | 3 | 0 | 0 |
| react_super | 5 | 1 | 0 | 0 |
| checklist | 5 | 3 | 3 | 0 |
| next_experiment | 5 | 1 | 0 | 0 |

## Cases

| Source record | Method | Case | Categories |
|---:|---|---|---|
| 1 | single_ultra | fault6-lr1e-5 | wrong_top_candidate, cause_not_confirmed |
| 2 | react_super | fault6-lr1e-5 | measured_fix_failed |
| 3 | checklist | fault6-lr1e-5 | wrong_confirmations, measured_fix_failed |
| 4 | next_experiment | fault6-lr1e-5 | measured_fix_failed |
| 9 | single_ultra | ref-fault6-mild-s0 | wrong_top_candidate, cause_not_confirmed |
| 11 | checklist | ref-fault6-mild-s0 | wrong_confirmations |
| 13 | single_ultra | ref-fault6-strong-s0 | cause_not_confirmed |
| 15 | checklist | ref-fault6-strong-s0 | wrong_confirmations |

Source record numbers refer to the normalized `rows.json` array (one-based).
Full score evidence is in `report.json`. Source file and truth hashes are recorded there.
This inventory does not establish why an error happened or evaluate new fault classes.
