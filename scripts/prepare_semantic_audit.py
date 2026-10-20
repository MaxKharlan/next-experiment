"""Freeze a controlled BANKING77 audit and a separate blind human review packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from core.eval.runtime import file_digest
from core.probes.level0.overlap import normalize
from scripts.fresh_answer_calibration import save

LABELS = (
    "card_arrival",
    "card_payment_fee_charged",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "declined_cash_withdrawal",
    "pending_cash_withdrawal",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)
SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv"


def prepare(source: Path, root: Path):
    if root.exists():
        raise FileExistsError("Frozen audit already exists")
    groups = {label: [] for label in LABELS}
    seen = set()
    with source.open() as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            key = normalize(row["text"])
            if row["category"] in groups and key not in seen:
                groups[row["category"]].append(
                    {"source_line": line, "text": row["text"], "label": row["category"]}
                )
                seen.add(key)
    reference, audit = [], []
    for label in LABELS:
        rows = sorted(
            groups[label],
            key=lambda r: hashlib.sha256(
                ("semantic-audit-20261020:" + r["text"]).encode()
            ).hexdigest(),
        )
        if len(rows) < 75:
            raise ValueError(f"Insufficient unique train rows for {label}")
        reference.extend(rows[:50])
        audit.extend(rows[50:75])
    random.Random(3101).shuffle(audit)
    planted = ["none"] * 100 + ["wrong_label"] * 50 + ["invalid_format"] * 50
    random.Random(3102).shuffle(planted)
    truth, public = [], []
    for i, (row, fault) in enumerate(zip(audit, planted, strict=True)):
        observed = row["label"]
        if fault == "wrong_label":
            observed = LABELS[(LABELS.index(observed) + 1 + i % 7) % 8]
        elif fault == "invalid_format":
            observed = json.dumps({"intent": observed})
        ident = f"audit-{i + 1:03d}"
        public.append({"id": ident, "text": row["text"], "observed_label": observed})
        truth.append(
            {
                "id": ident,
                "source_line": row["source_line"],
                "source_label": row["label"],
                "injection": fault,
            }
        )
    for i, row in enumerate(reference):
        row["id"] = f"reference-{i + 1:03d}"
    root.mkdir(parents=True)
    save(root / "reference.json", reference)
    save(root / "audit.json", public)
    save(root / "constructed-truth.json", truth)
    save(
        root / "human-review.json",
        {
            "reviewer": None,
            "reviewed_at": None,
            "labels": list(LABELS),
            "rows": [
                {
                    **row,
                    "format_error": None,
                    "contradiction": None,
                    "uncertain": None,
                    "correct_label": None,
                    "note": "",
                }
                for row in public
            ],
        },
    )
    protocol = {
        "id": "semantic-audit-20261020-v1",
        "source": SOURCE,
        "source_sha256": file_digest(source),
        "attribution": "PolyAI / Casanueva et al. (2020), BANKING77, CC BY 4.0. Selected train rows; planted modifications recorded separately.",
        "source_card": "https://huggingface.co/datasets/PolyAI/banking77",
        "labels": list(LABELS),
        "reference_examples": 400,
        "audit_examples": 200,
        "selection": "SHA256 seeded ordering within class; first 50 reference, next 25 audit; normalized exact texts disjoint. No test data.",
        "injections": {"none": 100, "wrong_label": 50, "invalid_format": 50},
        "embedding": {
            "local_qwen": "Qwen/Qwen3-Embedding-0.6B",
            "local_nvidia": "nvidia/Nemotron-3-Embed-1B-BF16",
            "provider": "Qwen/Qwen3-Embedding-8B",
            "representation": "Symmetric document embeddings, model document prompt, L2 normalized; no query instruction",
            "batch_size": 8,
            "max_length": 512,
        },
        "neighbor_rule": {
            "k": 3,
            "similarity_min": 0.8,
            "rule": "Flag when at least two eligible neighbors disagree with a valid observed label. Invalid format flagged separately.",
        },
        "near_duplicate_cosine_min": 0.92,
        "cleanlab": {
            "folds": 5,
            "shuffle_seed": 3103,
            "logistic_C": 1.0,
            "max_iter": 1000,
            "filter_by": "prune_by_noise_rate",
            "data": "400 references + valid audit labels, out-of-fold probabilities; invalid-format audit rows excluded",
        },
        "model_review": {
            "nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "super": "nvidia/nemotron-3-super-120b-a12b",
            "temperature": 0,
            "max_tokens": 2048,
            "attempts_per_example": 1,
            "context": "Same three Qwen local neighbors, query, observed label and allowed labels only. No source truth or injection type.",
        },
        "primary_human_metrics": None,
        "human_requirement": "Independent human completes all 200 rows, identity/date and uncertainty. Constructed-truth results are preliminary controls, not human accuracy. No adaptive threshold changes.",
        "source_code_sha256": file_digest(__file__),
        "data_sha256": {
            name: file_digest(root / name)
            for name in ("reference.json", "audit.json", "constructed-truth.json")
        },
    }
    save(root / "protocol.json", protocol)
    payload = json.dumps(
        json.loads((root / "human-review.json").read_text()), ensure_ascii=True
    ).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Independent audit · 200 examples</title>
<style>body{font:17px system-ui;max-width:950px;margin:36px auto;background:#faf9f6;color:#202020}article{padding:22px;background:white;margin:18px 0;border:1px solid #ccc;border-radius:8px}label{display:block;margin:10px 0}select,input,button{font:inherit;padding:6px}code{white-space:pre-wrap}header{position:sticky;top:0;background:#faf9f6;padding:12px 0}button{cursor:pointer}small{display:block}</style>
<h1>Independent label audit</h1><p>Review each query and the supplied label. The required format is exactly one allowed label, without JSON, explanation or punctuation. Contradiction means that the label conflicts with the query's intent; mark uncertain for ambiguous intent. Do not consult model outputs or constructed truth. Source: PolyAI BANKING77 (CC BY 4.0), selected and modified train examples.</p>
<details><summary>Allowed labels</summary><pre id="labels"></pre></details><header><label>Reviewer <input id="reviewer"></label><button id="export">Save review JSON</button> <span id="progress"></span><small>Answers are stored in this browser. Export the JSON when finished.</small></header><main id="rows"></main><script>
const initial=PAYLOAD, key='ne-semantic-audit-20261020-v1';let data=JSON.parse(localStorage.getItem(key)||'null')||initial;
const save=()=>{localStorage.setItem(key,JSON.stringify(data));document.querySelector('#progress').textContent=data.rows.filter(r=>['format_error','contradiction','uncertain'].every(k=>r[k]!==null)).length+'/200 reviewed'};
document.querySelector('#labels').textContent=data.labels.join('\\n');document.querySelector('#reviewer').value=data.reviewer||'';document.querySelector('#reviewer').oninput=e=>{data.reviewer=e.target.value;save()};
for(const r of data.rows){const a=document.createElement('article');let h=document.createElement('h2');h.textContent=r.id;a.append(h);let p=document.createElement('p');p.textContent=r.text;a.append(p);let c=document.createElement('code');c.textContent='Supplied label: '+r.observed_label;a.append(c);
for(const [k,title] of [['format_error','Format error'],['contradiction','Contradictory intent'],['uncertain','Uncertain intent']]){let l=document.createElement('label');l.textContent=title+' ';let s=document.createElement('select');for(const [v,t] of [['','Choose'],['true','Yes'],['false','No']]){let o=document.createElement('option');o.value=v;o.textContent=t;s.append(o)}s.value=r[k]===null?'':String(r[k]);s.onchange=()=>{r[k]=s.value===''?null:s.value==='true';save()};l.append(s);a.append(l)}
let l=document.createElement('label');l.textContent='Correct label (when identifiable) ';let s=document.createElement('select');for(const v of ['',...data.labels]){let o=document.createElement('option');o.value=v;o.textContent=v||'Choose';s.append(o)}s.value=r.correct_label||'';s.onchange=()=>{r.correct_label=s.value||null;save()};l.append(s);a.append(l);let n=document.createElement('input');n.placeholder='Reviewer note';n.value=r.note;n.oninput=()=>{r.note=n.value;save()};a.append(n);document.querySelector('#rows').append(a)}
document.querySelector('#export').onclick=()=>{data.reviewed_at=new Date().toISOString();save();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='human-review-completed.json';a.click();URL.revokeObjectURL(a.href)};save();</script>""".replace(
        "PAYLOAD", payload
    )
    (root / "human-review.html").write_text(html)
    print(
        json.dumps(
            {
                "root": str(root),
                "audit_rows": len(public),
                "reference_rows": len(reference),
                "protocol_sha256": file_digest(root / "protocol.json"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)
