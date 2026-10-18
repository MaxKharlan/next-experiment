# Saved benchmark failure inventory

Generated from saved score rows. Categories overlap; their counts are not additive.
Missing fix measurements are reported separately, not counted as failed fixes.

| Method | Cases | Cases with findings | Wrong confirmations | Confirmed without measured fix |
|---|---:|---:|---:|---:|
| single_ultra | 8 | 6 | 0 | 0 |
| react_super | 8 | 2 | 0 | 0 |
| checklist | 8 | 5 | 3 | 0 |
| next_experiment | 8 | 2 | 0 | 0 |

## Cases

| Source record | Method | Case | Categories |
|---:|---|---|---|
| 1 | single_ultra | ho-def-fault6-mild-s0 | cause_not_confirmed |
| 2 | react_super | ho-def-fault6-mild-s0 | measured_fix_failed |
| 3 | checklist | ho-def-fault6-mild-s0 | measured_fix_failed |
| 4 | next_experiment | ho-def-fault6-mild-s0 | measured_fix_failed |
| 5 | single_ultra | ho-def-fault6-strong-s0 | measured_fix_failed |
| 6 | react_super | ho-def-fault6-strong-s0 | measured_fix_failed |
| 7 | checklist | ho-def-fault6-strong-s0 | measured_fix_failed |
| 8 | next_experiment | ho-def-fault6-strong-s0 | measured_fix_failed |
| 9 | single_ultra | ho-e12-fault6-mild-s0 | wrong_top_candidate, cause_not_confirmed |
| 11 | checklist | ho-e12-fault6-mild-s0 | wrong_confirmations |
| 13 | single_ultra | ho-e12-fault6-strong-s0 | cause_not_confirmed |
| 15 | checklist | ho-e12-fault6-strong-s0 | wrong_confirmations |
| 21 | single_ultra | ho-e6-fault6-mild-s0 | cause_not_confirmed |
| 23 | checklist | ho-e6-fault6-mild-s0 | wrong_confirmations |
| 25 | single_ultra | ho-e6-fault6-strong-s0 | cause_not_confirmed |

Source record numbers refer to the normalized `rows.json` array (one-based).
Full score evidence is in `report.json`. Source file and truth hashes are recorded there.
This inventory does not establish why an error happened or evaluate new fault classes.
