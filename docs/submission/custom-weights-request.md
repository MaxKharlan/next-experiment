# Token Factory support request — ready to send

Destination: Token Factory support, through the console's Contact us action.
Status: not sent. After the owner asked whether custom weights was necessary,
the existing adapter-download/local-evaluation path was retained. Custom weights
is an optional alternative; it does not block the remaining queue.

Subject: Custom weights beta for evaluating Post-training LoRA checkpoints

Hello,

We are building Next Experiment for the Nebius x NVIDIA Global AI Hackathon. The tool investigates failed fine-tunes by comparing a proposed repair with a matched control. We currently train LoRA adapters through the Token Factory Post-training API and download result files for local evaluation.

Could you enable custom weights beta access for the current organization, or advise how to request it? Our intended use is short evaluation runs of Qwen3-0.6B and Qwen3-1.7B LoRA checkpoints, with fixed prompts and deterministic decoding. We need to preserve the base-model revision, tokenizer/chat template and adapter identity for each evaluation.

Please clarify whether Post-training result files can be used directly, which models and adapter formats are supported, how revisions and decoding settings are exposed, and how upload, serving and idle time are billed. We would also like to know how to unload or delete the evaluation endpoint and confirm that billing has stopped.

We can provide job IDs through the authenticated support channel if required. Thank you.
