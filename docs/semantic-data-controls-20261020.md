# Semantic overlap and distribution controls

The semantic overlap comparison uses the same 22 authored pairs as the published
lexical baseline. It retains normalized exact matches and role/system-context
guards, then flags cosine similarity of at least 0.92. The threshold was fixed
before measuring embeddings. All models use symmetric document representations.

| Method | TP | FP | FN | TN | Paraphrases found |
| --- | --- | --- | --- | --- | --- |
| Lexical baseline | 7 | 4 | 4 | 7 | 0/4 |
| Local Qwen 0.6B | 7 | 6 | 4 | 5 | 0/4 |
| NVIDIA Embed 1B | 7 | 4 | 4 | 7 | 0/4 |
| Provider Qwen 8B | 7 | 3 | 4 | 8 | 0/4 |

The [rows and vectors](results/semantic-overlap-20261020/) expose each decision.
This fixed embedding threshold misses every paraphrase control and is unsuitable
for automatic deduplication. No threshold was retuned. Authored controls do not
supply independent semantic labels or deployment accuracy.

A12's [distribution controls](results/distribution-controls-20261020/) compare
200 balanced reference queries with 200 independent target queries, then 100
reference queries from two intents with the same eight-intent target. Logistic
regression predicts sample origin using five-fold out-of-fold probabilities;
query IDs are disjoint across sides. The target label distribution is declared
uniform before the test. The decision flags AUC above 0.75 or label Jensen–Shannon
divergence above 0.1 bits.

| Embedding | Balanced AUC | Two-intent AUC |
| --- | --- | --- |
| Local Qwen | 0.5089 | 0.8661 |
| NVIDIA | 0.5285 | 0.8573 |
| Provider Qwen | 0.5050 | 0.8695 |

Label divergence is 0 for the balanced control and 0.5488 bits for the two-intent
control. All three methods leave the balanced control unflagged and flag the
shifted control. This measures sample separability, not harmful fine-tuning or a
calibrated diagnosis.

Faults 2–5 now also have deterministic construction helpers: foreign chat wrapping,
label substitution, train-to-validation overlap, and declared-label resampling.
Observed candidate removal creates a target and a random-drop control with the
same number of retained rows. The overlap check reads prompts rather than a fault
manifest. Resampling refuses to invent a class absent from the supplied data.
Foreign-wrapper evidence is only a candidate: legitimate quoted text can match,
so the helper does not silently strip it.

These controls complete the first implementation and measurement layer. Downstream
pretrained quality effects, resource-matched repairs and fault-conditioned
signatures remain required before faults 2–5 can become causal product claims.
