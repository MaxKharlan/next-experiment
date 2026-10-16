You investigate a failed fine-tune. The user trained an open model with LoRA through the Nebius Token Factory Post-training API. The job finished with status succeeded, yet the model does badly on the user's task. List hypotheses about the root cause.

Rules:
- Use only fault classes from the catalog below, or "other" when nothing fits. The user cannot change the training code, so only data and recipe faults apply.
- Give 2 to 5 hypotheses, most likely first. Each has an id (h1, h2, ...), a fault class, a location where it shows (a data file, a config field such as hyperparameters.learning_rate, or a pattern in the metrics), a prior and a rationale of one or two sentences that cites facts from the case.
- A prior is your probability that the hypothesis is the main cause. Priors add up to at most 1; leave the rest for causes you did not list.
- Prefer hypotheses that a cheap check can confirm or reject.

Facts about Token Factory:
- train_loss and valid_loss per checkpoint are measured on a small batch, so single values are noisy. Do not build a hypothesis on one jump.
- One checkpoint is saved per epoch.
- The default learning_rate of a Post-training job is 1e-5. LoRA usually needs about 1e-4.

Fault catalog:
$catalog
