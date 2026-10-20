# Next Experiment

An agent that investigates failed LLM fine-tunes with controlled experiments.

## Inspiration

A training job can finish successfully while the resulting model still fails its task. A loss curve does not tell an engineer whether to change the data, the recipe or the trainer. We built Next Experiment to turn that debugging loop into a recorded sequence of hypotheses, checks and comparisons with explicit costs.

## What it does

Next Experiment inspects a fine-tuning case, ranks possible causes and chooses checks by their expected information per dollar. A proposed repair is compared with a declared control on separate check examples. The report records the effect, uncertainty, decision rule and cost. Inconclusive results remain inconclusive. An approved final retrain is evaluated alongside the base and failed models.

The public application offers live Nemotron investigations on uploaded chat datasets, without an API key or payment from testers. A shared $20 allocation covers these additional runs, with the remaining budget shown before starting. The hosted runner supports Qwen3-0.6B, short-answer exact match and one target/control comparison with two seeds per arm. Uploaded recipes are unverified, and small-sample findings are exploratory. Local training-label checks and recorded replays remain available without spending this allocation. In that historical case, Qwen3-0.6B improved from 14% to 42.5% exact match on 200 held-out queries. Estimated inference and training cost was about $0.075. This result concerns one task and recipe, not a general recovery rate.

## How we built it

The application uses Python, FastAPI and a browser interface with server-sent events. NVIDIA Nemotron 3 Super ranks hypotheses; Ultra explains supported verdicts. Token Factory provides inference and LoRA training through an OpenAI-compatible client. Tavily supplies linked reports of known issues. Local PyTorch evaluates downloaded adapters.

State, execution ownership and budgets use SQLite. Every paid operation reserves its estimated maximum cost before dispatch; uncertain outcomes retain their reservation. Reports store available provider request IDs. Artifact storage supports local files and Nebius Object Storage. A pinned base checkpoint produced identical records on 64 development queries across MacBook MPS, AI Cloud L40S and Kaggle T4. Equivalence of trained adapters across those backends remains unverified.

A fresh REST investigation completed eight local adapter runs, saved its event stream and report, and verified recovery after an API restart. Repeating its POST reused the same investigation. Its diagnosis remained inconclusive.

We separately measured Nano on 200 constructed semantic-audit examples. Its failure to flag the planted label contradictions is published alongside stronger embedding and cleanlab baselines. Those measurements are not presented as a calibrated live classifier.

## Challenges

Training seed labels did not guarantee independent Token Factory adapters. We now inspect weight hashes and record order changes explicitly. We also found that adapter parameter names needed normalization before local loading. Ignoring that mismatch would have evaluated unloaded weights.

A repair beating its control does not prove that it is the unique root cause or that the model has recovered to a healthy reference. Our reports separate those claims and preserve rejected development designs.

## Accomplishments

The historical benchmark contains 13 cases and four methods. On its nine low-learning-rate failures, Next Experiment ranked the planted cause first in all nine and recovered six; it raised no false alarms on the four healthy cases. Average estimated cost was $0.144 per case versus $0.205 for the ReAct baseline. The benchmark covers a narrow failure family, and its exposed cases are available for inspection rather than reuse as fresh held-out data.

FT-Fail is published on Hugging Face with the 13 cases, 52 saved method results, 200 automatic audit examples and 400 reference examples. The package includes hashes, provenance and source licenses.

A separate Qwen reference study verified 5,012 saved answers on AI Cloud. Adapters selected on validation scored 104/154 for 0.6B and 115/154 for 1.7B on Banking77; 100/102 and 102/102 on function calling; and 102/102 for both on JSON extraction. These are development results on previously inspected structured splits. The 0.6B base structured answers used Markdown fences and failed the strict JSON contract. The 1.7B base already scored 100% on both structured tasks, with no improvement from its selected adapters.

## What we learned

A strong model can often name a likely cause without an experiment. Experiments matter when deciding whether the proposed change actually helps. Negative results also matter: some planted faults did not reduce quality, and a classifier that looked useful on local runs failed to transfer reliably to provider runs.

## What's next

Extend the completed Qwen clean references to missing model/fault comparisons and calibrate diagnostic probabilities on independently sampled failures. The Cloud study completed all fifteen code comparisons and 29 of 36 data comparisons before its frozen timeout; seven cells remain unmeasured. Broader deployment needs evidence beyond the existing development tasks.

## Links and track

Recommended track: Coding and Agentic Engineering.

- [Live testing and recorded investigations](https://maxkharlan.github.io/next-experiment/)
- [Source and setup](https://github.com/MaxKharlan/next-experiment)
- [FT-Fail dataset](https://huggingface.co/datasets/MakHH/FT-Fail)
- [Detailed benchmark](https://github.com/MaxKharlan/next-experiment/blob/main/docs/benchmark.md)
