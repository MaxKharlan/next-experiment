"""Check Token Factory access for the three Nemotron models from .env."""

from openai import OpenAI

from core.config import get_settings

s = get_settings()
client = OpenAI(api_key=s.token_factory_api_key, base_url=s.token_factory_base_url)

for name, model in [
    ("nano", s.nemotron_nano_model),
    ("super", s.nemotron_super_model),
    ("ultra", s.nemotron_ultra_model),
]:
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=256,
    )
    choice = reply.choices[0]
    print(
        f"{name:5} {model} -> {choice.message.content!r} "
        f"({choice.finish_reason}, {reply.usage.total_tokens} tokens)"
    )
