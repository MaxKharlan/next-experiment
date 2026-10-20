"""Repair the explicitly declared role-line producer, never arbitrary quoted text."""

import copy
import re

from bench.faults.semantic_data import FOREIGN_PREFIX, FOREIGN_SUFFIX, template_evidence

PRODUCER = "role-lines-inst-wrapper-v1"


def repair_declared_wrapper(rows, *, producer):
    if producer != PRODUCER:
        raise ValueError("Repair requires the exact declared producer format")
    out = copy.deepcopy(rows)
    candidates = template_evidence(rows)["candidate_indices"]
    for index in candidates:
        wrapper, answer = out[index]["messages"]
        body = wrapper["content"][len(FOREIGN_PREFIX) : -len(FOREIGN_SUFFIX)]
        matches = list(re.finditer(r"(?m)^(system|user|assistant): ", body))
        if not matches or matches[0].start() != 0:
            raise ValueError("Malformed declared role-line wrapper")
        roles = [match[1] for match in matches]
        if roles not in (["user"], ["system", "user"]):
            raise ValueError("Ambiguous role markers in declared wrapper")
        restored = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() - 1 if i + 1 < len(matches) else len(body)
            text = body[match.end() : end]
            if not text.strip():
                raise ValueError("Empty message in declared wrapper")
            restored.append({"role": match[1], "content": text})
        out[index]["messages"] = [*restored, answer]
    return out, candidates
