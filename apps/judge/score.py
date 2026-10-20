"""Score one adapter in a child process so its model memory is fully released."""

import json
import sys
from pathlib import Path

from core.case import EvaluatorSpec
from core.eval import evaluate
from core.eval.runtime import file_digest


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    result = evaluate(
        request["data"],
        EvaluatorSpec.model_validate(request["evaluator"]),
        adapter=request["adapter"],
        base_model="Qwen/Qwen3-0.6B",
        device="cpu",
    )
    saved = result.to_dict()
    saved["adapter_files_sha256"] = {
        name: file_digest(Path(request["adapter"]) / name)
        for name in ("adapter_model.safetensors", "adapter_config.json")
    }
    Path(request["output"]).write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
