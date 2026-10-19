/* Browser port of core/agent/label_checks.py and bench/faults/labels.py. */
(function (root) {
  'use strict';
  const fail = message => { throw new Error(message); };
  // Preserve integer precision and distinguish JSON 1 from 1.0, as StrictInt does.
  function parse(text) {
    let i = 0;
    const ws = () => { while (/[ \t\r\n]/.test(text[i] || '') && i < text.length) i++; };
    function value() {
      ws(); const start = i, c = text[i];
      if (c === '"') {
        i++;
        while (i < text.length) { if (text[i++] === '\\') i++; else if (text[i - 1] === '"') return JSON.parse(text.slice(start, i)); }
        fail('Unterminated JSON string');
      }
      if (c === '[' || c === '{') {
        i++; ws(); const array = c === '[', end = array ? ']' : '}', out = array ? [] : Object.create(null);
        if (text[i] === end) { i++; return out; }
        while (true) {
          if (array) out.push(value());
          else { ws(); if (text[i] !== '"') fail('Expected a JSON property name'); const key = value(); ws(); if (text[i++] !== ':') fail('Expected colon'); out[key] = value(); }
          ws(); const next = text[i++]; if (next === end) return out; if (next !== ',') fail('Expected comma');
        }
      }
      const literal = /^(true|false|null)/.exec(text.slice(i));
      if (literal) { i += literal[0].length; return JSON.parse(literal[0]); }
      const number = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(i));
      if (!number) fail('Invalid JSON value');
      i += number[0].length;
      if (/[.eE]/.test(number[0])) return {invalidFloat: number[0]};
      const n = BigInt(number[0]); return n <= BigInt(Number.MAX_SAFE_INTEGER) && n >= BigInt(Number.MIN_SAFE_INTEGER) ? Number(n) : n;
    }
    const result = value(); ws(); if (i !== text.length) fail('Unexpected JSON content'); return result;
  }
  const integer = x => typeof x === 'bigint' || Number.isSafeInteger(x);
  function object(x, keys, path) {
    if (!x || typeof x !== 'object' || Array.isArray(x)) fail(`${path} must be an object`);
    for (const key of Object.keys(x)) if (!keys.includes(key)) fail(`${path}: unknown field ${key}`);
  }
  function checkTrainingLabels(data) {
    object(data, ['reference', 'observed', 'eos_token_id'], 'training_labels');
    let positions = 0;
    for (const side of ['reference', 'observed']) {
      const rows = data[side];
      if (!Array.isArray(rows) || rows.length < 1 || rows.length > 1024) fail(`${side} must contain 1–1024 rows`);
      rows.forEach((row, index) => {
        const path = `${side}[${index}]`; object(row, ['input_ids', 'labels'], path);
        for (const field of ['input_ids', 'labels']) {
          if (!Array.isArray(row[field]) || row[field].length > 32768) fail(`${path}.${field} must be an array of at most 32768 integers`);
          if (row[field].some(x => !integer(x) || (x < 0 && !(field === 'labels' && x === -100)))) fail(`${path}.${field}: expected nonnegative integer token IDs${field === 'labels' ? ' or -100' : ''}`);
        }
        if (row.input_ids.length !== row.labels.length) fail('Input IDs and labels must have equal lengths');
        positions += row.input_ids.length;
        if (side === 'reference' && row.labels.some((x, j) => x !== -100 && x !== row.input_ids[j])) fail('Reference labels must be unshifted token IDs or -100');
      });
    }
    if (positions > 131072) fail('Label checks allow at most 131072 token positions');
    const eosId = data.eos_token_id ?? null;
    if (eosId !== null && (!integer(eosId) || eosId < 0)) fail('eos_token_id must be a nonnegative integer or null');
    if (data.reference.length !== data.observed.length) fail('Reference and observed rows must align');
    let healthy = 0, shifted = 0, informative = 0, targets = 0;
    let prompt = 0, included = 0, answer = 0, promptOK = true;
    let before = 0, after = 0, eosTargets = 0, removed = 0, eosOK = true;
    data.reference.forEach((ref, r) => {
      const obs = data.observed[r];
      if (ref.input_ids.length !== obs.input_ids.length || ref.input_ids.some((x, j) => x !== obs.input_ids[j])) fail('Reference and observed tokens must align');
      for (let j = 1; j < ref.labels.length; j++) {
        const expected = ref.labels[j], actual = obs.labels[j], token = ref.input_ids[j], faulty = ref.labels[j + 1] ?? -100;
        healthy += actual !== expected; shifted += actual !== faulty; informative += expected !== faulty; targets += actual !== -100;
        if (expected === -100) { prompt++; included += actual !== -100; if (actual !== -100 && actual !== token) promptOK = false; }
        else { answer++; if (actual !== expected) promptOK = false; }
        if (actual !== expected && actual !== -100) eosOK = false;
        before += expected !== -100; after += actual !== -100;
        if (expected === eosId) { eosTargets++; removed += actual === -100; }
      }
    });
    return {
      scope: 'supplied unpadded labels; reference correctness is caller-owned', rows: data.reference.length,
      alignment: {status: !informative ? 'inconclusive' : !healthy ? 'aligned' : !shifted ? 'double_shift' : 'other_mismatch', informative_positions: informative, healthy_mismatches: healthy, preshift_mismatches: shifted, observed_targets: targets},
      prompt: promptOK ? {prompt_positions: prompt, supervised_prompt_tokens: included, answer_targets: answer, total_targets: answer + included, prompt_loss_fraction: answer + included ? included / (answer + included) : null, status: included ? 'included' : prompt ? 'masked' : 'no_prompt_targets'} : {status: 'not_applicable', reason: 'labels are not prompt-only additions'},
      eos: eosId === null ? {status: 'unavailable', reason: 'EOS token ID was not supplied'} : !eosOK ? {status: 'not_applicable', reason: 'labels are not masking-only changes'} : {eos_token_id: eosId, reference_targets: before, observed_targets: after, reference_eos_targets: eosTargets, removed_eos_targets: removed, removed_eos_fraction: eosTargets ? removed / eosTargets : null, status: !eosTargets ? 'no_eos_targets' : removed ? 'removed' : 'preserved'},
      causal_confirmation: false
    };
  }
  function request(text) {
    const data = parse(text); object(data, ['title', 'training_labels'], 'request');
    if (data.title !== undefined && (typeof data.title !== 'string' || [...data.title].length < 1 || [...data.title].length > 120)) fail('title must contain 1–120 characters');
    return checkTrainingLabels(data.training_labels);
  }
  function stringify(value, depth = 0) {
    if (typeof value === 'bigint') return String(value);
    if (!value || typeof value !== 'object') return JSON.stringify(value);
    const indent = '  '.repeat(depth), entries = Object.entries(value).map(([key, v]) => `${indent}  ${JSON.stringify(key)}: ${stringify(v, depth + 1)}`);
    return `{\n${entries.join(',\n')}\n${indent}}`;
  }
  const api = {parse, request, checkTrainingLabels, stringify};
  if (typeof module !== 'undefined') module.exports = api;
  else root.LabelAudit = api;
})(globalThis);
