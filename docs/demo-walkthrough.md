# O2 demo walkthrough

Build the public replays without credentials or training:

```bash
uv run python -m scripts.build_measured_demos
uv run python -m scripts.build_pages --out _site
python -m http.server --bind 127.0.0.1 --directory _site
```

Open localhost:8000 and choose **Answer cuts on Token Factory — measured
experiment**, then Replay. Explain that an upstream script damaged answers before
upload; the provider itself rejected over-limit examples in the recorded test.
The four arms show healthy 92.6%, damaged 68.1%, target 85.9% and control 66.2%
on 770 check examples. The target removes flagged examples; the same-size control
removes random unflagged examples. Show the two-seed caveat and original job IDs,
train hashes and adapter hashes in the sources disclosure.

Choose the Kaggle experiment. Its three-seed comparison has the same arm ordering,
with lower absolute quality. The noise calculation is local to those runs.
Both screens say recovery is partial and identify the replay as an experiment.
No Super hypotheses, Ultra verdict or API investigation trace is invented.

For the agent workflow, choose the historical **Fault 6 on Token Factory** run.
Its stored `confirmed` status means a useful learning-rate intervention against
its control. Explain that uniqueness of the cause requires tests of competing
explanations. This historical run is separate from the answer-cut experiments.

Switch to a local audit to demonstrate that the previous result clears before
Replay. A label or dataset audit shows observations without claiming a measured
quality failure.

O2 still needs a fresh complete answer-cut API investigation with persisted
hypotheses, probe outputs, approval, intervention and final report; a public demo
video and submission packaging are also pending. These replays do not close those
requirements. Any paid run needs its explicit budget and a frozen evaluation plan.
