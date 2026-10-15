"""Check that the three Nemotron models return valid structured output via the connector."""

from core.agent.schemas import HypothesisList
from core.connectors.token_factory import JSON_FORMATS, TokenFactory

PROMPT = (
    "A Qwen3-0.6B LoRA fine-tune finished with status succeeded, but a third of its "
    "answers on long inputs are cut-off JSON. Propose two hypotheses."
)

tf = TokenFactory.from_settings()
for role in ("nano", "super", "ultra"):
    model = tf.model(role)
    result, calls = tf.structured(
        model, [{"role": "user", "content": PROMPT}], HypothesisList
    )
    classes = [h.fault_class for h in result.hypotheses]
    cost = sum(c.cost_usd for c in calls)
    fmt = JSON_FORMATS[tf.format_level[model]]
    print(f"{role:5} {fmt:11} {len(calls)} call(s) ${cost:.5f} {classes}")
