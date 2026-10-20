"""Measure the deployed CPU runtime on public synthetic prompts, without paid calls."""

import json
import resource
import time

from core.eval.data import Example
from core.eval.generate import answer, load_base, load_tokenizer
from core.eval.metrics import get_metric


def main():
    from core.agent.steps import load_prompt
    from core.interventions.answer_check import investigate_with_answers

    assert callable(investigate_with_answers)
    assert load_prompt("hypotheses").text
    started = time.monotonic()
    model = load_base("Qwen/Qwen3-0.6B")
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")
    model.eval()
    result = answer(
        model,
        tokenizer,
        Example(
            line=1,
            prompt=[
                {
                    "role": "user",
                    "content": "Read this context: "
                    + "x " * 1800
                    + "Reply with just the word ready.",
                }
            ],
            expected="ready",
        ),
        "cpu",
        get_metric("exact_match"),
        32,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "lora_rank": 64,
                "context_repetitions": 1800,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "seconds": time.monotonic() - started,
                "answer": result.answer,
            }
        )
    )


if __name__ == "__main__":
    main()
