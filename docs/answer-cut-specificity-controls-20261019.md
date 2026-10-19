# Answer-cut specificity controls — 19 October 2026

`answer-cut-v2` produces identical observations on some clean labels and genuinely
truncated labels. It also flags internal JSON syntax errors as `cut`. These
controls measure two obstacles to treating that signal as a unique cause.

Protocol `dad057663b78a7f21716d31b5790be5983248bec3da85fd9f3b873118a3375a0`
was fixed before execution. The unmodified probe ran on 35 synthetic development
files, each with 100 rows: seven classes at affected counts 1, 5, 10, 25 and 50.
For clean-prefix cases those counts denote valid short labels, not damage.
No model was trained or evaluated and no provider calls were made.

| Constructed class | `cut` outcomes at counts 1 / 5 / 10 / 25 / 50 | Incorrectly flagged clean rows, summed across the five files | Missed damaged rows |
| --- | --- | ---: | ---: |
| Clean labels without prefix relationships | no / no / no / no / no | 0 | 0 |
| Valid `pay` label beside `pay_later` | yes / yes / yes / yes / no | 41 | 0 |
| `pay_later` truncated to `pay` | yes / yes / yes / yes / no | 0 | 50 |
| Clean JSON | no / no / no / no / no | 0 | 0 |
| JSON with its suffix removed | yes / yes / yes / yes / no | 0 | 50 |
| JSON with an internal comma removed | yes / yes / yes / yes / no | 0 | 50 |
| Valid JSON containing a wrong value | no / no / no / no / no | 0 | 91 |

The clean-prefix and truncated-prefix fixtures have the same observed answer
sequence at every count. Their user requests differ: one explicitly asks for
`pay`, the other asks for `pay_later`. The probe reads only answers and returns
identical complete raw results for each paired fixture. At counts 1–25, dropping
flagged rows would remove every valid `pay` example from the clean-prefix file.
The effect on a trained model was not measured.

The JSON syntax control removes an internal comma while keeping the final
closing brace. It is damaged data, but it is not suffix truncation. At 50%
malformed JSON, the probe's strict-majority anchor switches off. Valid wrong
values parse successfully and escape this structural probe.

## Calibration consequence

The current signature table assigns `P(cut | answer_truncation) = 0.90` and
`P(cut | default) = 0.03`, a likelihood ratio of 30. These are heuristic values.
This constructed coverage matrix cannot estimate deployment frequencies, so no
new population likelihood or confidence interval is fitted from its counts.
The results require clean-prefix and alternative-malformation cases in any
future calibration dataset. They also separate observed structural defects from
cause identification and model-quality damage.

T=1, the signature table and all live probe behavior remain unchanged. This
closes the requested construction and measurement of clean-prefix development
controls. Fresh causal target/control evidence, benign recipe controls and
independent signature calibration remain open.

## Artifacts and reproduction

[Protocol and fixtures](results/answer-cut-specificity-controls-20261019/protocol.json)
record the source hashes, exact inputs and changed line numbers.
[Results](results/answer-cut-specificity-controls-20261019/results.json) preserve
all per-case probe outputs and the paired comparisons.

```bash
uv run python -m bench.answer_cut_controls prepare \
  --root artifacts/answer-cut-specificity-reproduction
uv run python -m bench.answer_cut_controls run \
  --root artifacts/answer-cut-specificity-reproduction
```

The protocol and result JSON reproduce byte-for-byte in a new directory.
The fixtures are a deterministic coverage matrix, not independent samples from
a population or repeated training seeds.
