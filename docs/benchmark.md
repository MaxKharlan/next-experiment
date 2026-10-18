# Benchmark: dev and held-out runs

This is the benchmark of section 10: four diagnosis methods on fine-tuning jobs whose cause of failure is known, scored by the same rules. The runner is `bench/run_bench.py`, the metrics are defined in `bench/score.py`, and the rows behind every table are in `artifacts/bench/<split>/rows.jsonl`.

## Setup

Every case is a Token Factory job on Qwen3-0.6B that learns to answer BANKING77 queries with one of 77 intent labels, trained on 200 examples. The metric is exact match on 200 held-out queries. The reference, the best recipe found for this task and model (B5), scores 0.535.

The dev split has 5 cases and the held-out split has 8. Healthy cases have no planted fault, so the right answer there is to confirm nothing.

| Split | Case | Recipe | Planted fault | Exact match of the job |
|---|---|---|---|---|
| dev | fault6-lr1e-5 | 3 epochs, rank 16 | learning rate 1e-5 | 0.140 |
| dev | ref-fault6-mild-s0 | 6 epochs, rank 16 | learning rate 2e-5 | 0.355 |
| dev | ref-fault6-strong-s0 | 6 epochs, rank 16 | learning rate 6e-6 | 0.205 |
| dev | ref-healthy-s0 | 6 epochs, learning rate 2e-4 | none | 0.535 |
| dev | ref-epochs18-s0 | 18 epochs, learning rate 2e-4 | none | 0.560 |
| held-out | ho-def-fault6-mild-s0 | Token Factory defaults: 3 epochs, rank 8, alpha 8 | learning rate 1e-5 | 0.080 |
| held-out | ho-def-fault6-strong-s0 | Token Factory defaults | learning rate 3e-6 | 0.035 |
| held-out | ho-e6-fault6-mild-s0 | 6 epochs, rank 16 | learning rate 1e-5 | 0.260 |
| held-out | ho-e6-fault6-strong-s0 | 6 epochs, rank 16 | learning rate 3e-6 | 0.085 |
| held-out | ho-e6-healthy-s0 | 6 epochs, learning rate 1e-4 | none | 0.500 |
| held-out | ho-e12-fault6-mild-s0 | 12 epochs, rank 16 | learning rate 2e-5 | 0.420 |
| held-out | ho-e12-fault6-strong-s0 | 12 epochs, rank 16 | learning rate 6e-6 | 0.270 |
| held-out | ho-e12-healthy-s0 | 12 epochs, learning rate 2e-4 | none | 0.555 |

The held-out cases come from fault sweeps around recipes the dev split does not have, and they were built before any method ran on them (`scripts/make_case.py`).

Next Experiment is the agent. Nemotron Super proposes causes with probabilities, the planner picks the probe or causal check with the most expected information per dollar, and a cause counts as proved only when a causal check confirms it.

The three methods it is compared with are in `bench/baselines/`. ReAct gives Super the same tools, the same causal check and the same limit on checks, and lets Super choose the order instead of the planner. The checklist runs every free probe, then a causal check for every fault class that has one, whatever the probes said; when more than one check confirms, the largest effect wins, and no model is called. Single Ultra is one call to Nemotron Ultra with the whole case and no experiments.

A causal check trains the fix and a control of the same cost on Token Factory, two seeds each, and scores the adapters on a separate 200-query check set (`artifacts/b10/check.jsonl`). A fix counts as a cause when it beats the control by at least 0.06, about three times the noise between seeds (SD 0.019, `scripts/rescore_pairs.py`). That threshold, the epoch-count probe's threshold and the lowest probability at which a hypothesis gets a check were set on the dev run, before the held-out run.

Fix success retrains the case's recipe with the standard fix for the confirmed cause on the whole training file. It succeeds when the result reaches the reference within 0.05, so at least 0.485.

Costs are LLM tokens at list price plus estimated training: training tokens times Token Factory's price per million tokens. Each method pays for every check it asked for, even when another method had already trained that pair and the cache served it.

## Results

### Dev, 5 cases (3 with a fault, 2 healthy)

| method | top-1 | top-3 | false alarms | wrong confirmations | fix success | P(true cause) | $ per case | jobs per case |
|---|---|---|---|---|---|---|---|---|
| Single Ultra | 1/3 | 2/3 | 0/2 | 0 | 0/3 | 0.28 | 0.011 | 0.0 |
| ReAct | 3/3 | 3/3 | 0/2 | 0 | 2/3 | 0.87 | 0.238 | 5.6 |
| Checklist | 3/3 | 3/3 | 0/2 | 3 | 2/3 | 0.14 | 0.427 | 12.0 |
| Next Experiment | 3/3 | 3/3 | 0/2 | 0 | 2/3 | 0.61 | 0.126 | 3.2 |

### Held-out, 8 cases (6 with a fault, 2 healthy)

| method | top-1 | top-3 | false alarms | wrong confirmations | fix success | P(true cause) | $ per case | jobs per case |
|---|---|---|---|---|---|---|---|---|
| Single Ultra | 5/6 | 5/6 | 0/2 | 0 | 0/6 | 0.46 | 0.008 | 0.0 |
| ReAct | 6/6 | 6/6 | 0/2 | 0 | 4/6 | 0.84 | 0.185 | 4.5 |
| Checklist | 6/6 | 6/6 | 0/2 | 3 | 4/6 | 0.23 | 0.410 | 12.0 |
| Next Experiment | 6/6 | 6/6 | 0/2 | 0 | 4/6 | 0.63 | 0.156 | 4.0 |

The held-out run ran once. It stopped on the fourth case when Token Factory refused new files past its limit of 5000, and continued with `--resume`, which scores the diagnoses already saved instead of asking the models again.

### Both splits, 13 cases

| method | top-1 | false alarms | wrong confirmations | fix success | $ per case | jobs per case | $ on the 4 healthy cases |
|---|---|---|---|---|---|---|---|
| Single Ultra | 6/9 | 0/4 | 0 | 0/9 | 0.009 | 0.0 | 0.035 |
| ReAct | 9/9 | 0/4 | 0 | 6/9 | 0.205 | 4.9 | 1.358 |
| Checklist | 9/9 | 0/4 | 6 | 6/9 | 0.417 | 12.0 | 2.298 |
| Next Experiment | 9/9 | 0/4 | 0 | 6/9 | 0.144 | 3.7 | 0.565 |

Next Experiment and ReAct find the same causes and make the same mistakes, none. Next Experiment costs 30% less per case. The checklist finds the cause too, at 2.9 times the cost of Next Experiment, and proves a second, wrong cause on 6 of the 9 fault cases.

The saving comes from the healthy cases. On the fault cases the two agents spent the same: $0.294 each on dev, $1.018 and $1.017 on held-out. On the 4 healthy cases Next Experiment spent $0.565 and ReAct $1.358.

### Held-out, case by case

| Case | Single Ultra | ReAct | Checklist | Next Experiment |
|---|---|---|---|---|
| ho-def-fault6-mild-s0 | no cause | lr_too_low, 4 jobs, $0.060 | lr_too_low, 12 jobs, $0.164 | lr_too_low, 4 jobs, $0.059 |
| ho-def-fault6-strong-s0 | lr_too_low | lr_too_low, 4 jobs, $0.058 | lr_too_low, 12 jobs, $0.164 | lr_too_low, 4 jobs, $0.060 |
| ho-e6-fault6-mild-s0 | no cause | lr_too_low, 4 jobs, $0.113 | lr_too_low and lora_rank_too_small, 12 jobs, $0.328 | lr_too_low, 4 jobs, $0.116 |
| ho-e6-fault6-strong-s0 | no cause | lr_too_low, 4 jobs, $0.113 | lr_too_low, 12 jobs, $0.328 | lr_too_low, 4 jobs, $0.115 |
| ho-e6-healthy-s0 | no cause | no cause, 0 jobs, $0.008 | no cause, 12 jobs, $0.328 | no cause, 0 jobs, $0.005 |
| ho-e12-fault6-mild-s0 | no cause | lr_too_low, 8 jobs, $0.449 | lr_too_low and lora_rank_too_small, 12 jobs, $0.657 | lr_too_low, 8 jobs, $0.445 |
| ho-e12-fault6-strong-s0 | no cause | lr_too_low, 4 jobs, $0.223 | lr_too_low and lora_rank_too_small, 12 jobs, $0.657 | lr_too_low, 4 jobs, $0.224 |
| ho-e12-healthy-s0 | no cause | no cause, 8 jobs, $0.452 | no cause, 12 jobs, $0.657 | no cause, 4 jobs, $0.224 |

A check on an e12 case costs four times one on a def case because the jobs train for 12 epochs instead of 3.

## Error analysis

### The checklist proves a second cause when the learning rate is too low

All 6 of its wrong confirmations are lora_rank_too_small on a case whose true cause is lr_too_low: the 3 dev fault cases and 3 of the 6 held-out ones. The rank check trains a larger adapter, and a larger adapter also learns faster when the learning rate holds the job back. On held-out it beat the control by 0.075 on e12-mild, 0.135 on e12-strong and 0.133 on e6-mild, all above the 0.06 threshold. It stayed below on e6-strong (0.040), def-mild (0.058) and def-strong (0.020). A causal check shows that a change helps; it cannot show that the change acts on the cause. A method that runs every check will find every fix that helps. The two agents stop at the first confirmed cause, and on every fault case they checked the learning rate before rank: first on 8 of the 9, second on e12-mild after epochs. So neither ever ran the rank check on a fault case. Had one tested rank first, it would have confirmed rank and stopped.

### On ho-e12-fault6-mild both agents paid for a check the free probes could have saved

Super's hypotheses ranked too many epochs first (0.40), then LoRA rank (0.35), then the learning rate (0.15), which is a fair reading of a 12-epoch job. The loss-curve probe then reported overfitting. Next Experiment trained the epochs check (fewer epochs lost 0.165, rejected) before it ran the free learning-rate probe, whose expected gain was 0.19 bits; that probe said the learning rate was low and the next check confirmed it. ReAct ran the learning-rate probe early but still tested epochs first. Each spent 4 extra jobs, about $0.22.

### On healthy cases Next Experiment stops sooner

On ho-e6-healthy both agents stopped after the free probes, without training. On ho-e12-healthy Next Experiment ran one check (fewer epochs, effect −0.003, rejected) and stopped, while ReAct also checked rank. On dev ref-epochs18 it was 4 jobs against 8. On dev ref-healthy Next Experiment trained nothing, and ReAct ran a learning-rate check and a rank check that both came back inconclusive. The checklist trains 12 jobs on every healthy case by design. No method raised a false alarm.

### Fix success is 6 of 9 for every method that confirms

The three failures are the same cases for all of them: the two Token Factory default recipes, where the learning-rate fix scores 0.395, and dev fault6-lr1e-5, where it scores 0.425. Both are below 0.485. The reference trains for 6 epochs at 2e-4; a 3-epoch recipe with only its learning rate fixed stays short of it. The standard fix changes one factor, so these failures measure the case, not the method.

### P(true cause) is not a measure of being right here

Every agent that confirmed named the true cause. Next Experiment reports the planner's belief after its probes; a confirmed causal check decides the verdict but does not raise that number, which averages 0.62. ReAct reports Super's own confidence, which is higher. The checklist reports a spread close to uniform.

### Single Ultra ranks well but commits rarely

Its first cause was the true one in 5 of 6 held-out cases but only 1 of 3 on dev, and it confirmed a cause once in 9 fault cases. That one was ho-def-fault6-strong, where the fix fails for every method, so its fix success is 0 of 9.

## Limits

- Every fault case has the same true cause, a learning rate that is too low. In the B9 sweeps the other planted faults did not hurt this task: LoRA rank 4 or 2 scored like rank 16, and more epochs helped. So the benchmark tests whether a method finds the cause under different recipes and stays quiet on healthy jobs. It does not test whether a method can tell two real causes apart.
- One task, one model, two seeds per arm and 13 cases. The rates are counts, and one case changes them by a lot.
- Each split ran once. The LLM methods are not deterministic, so a rerun can choose other checks and spend differently.

## Changes these results suggest

None of these were applied to the numbers above.

- Run the free probes before any paid check. On ho-e12-fault6-mild this would have saved 4 jobs for both agents.
- Update the belief after a causal check, so P(true cause) reflects the experiment as well as the probes.
- Make the rank check separate capacity from learning speed, or compare effect sizes when two checks confirm, so a second fix that helps for the same reason is not reported as a second cause.

## Rerunning

Dev, from the cached pairs and fixes, without training:

    uv run --group ml --group eval python -m bench.run_bench --split dev --no-train --check-set artifacts/b10/check.jsonl --noise-sd 0.019 --min-effect 0.06

Held-out, with training on Token Factory. Keep `scripts/tf_prune_checkpoints.py --delete --watch 10` running next to it so the account stays under its file limit:

    caffeinate -i uv run --group ml --group eval python -m bench.run_bench --split heldout --check-set artifacts/b10/check.jsonl --noise-sd 0.019 --min-effect 0.06 --resume
