"""Run the screen's cost rendering against event fixtures in Node."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_screen_records_retries_without_double_counting():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the browser JavaScript")
    page = Path(__file__).resolve().parents[2] / "apps/api/static/index.html"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(source);
const nodes = {};
const context = {
  $: id => nodes[id] ||= {style: {}, hidden: true},
  esc: x => String(x).replaceAll('<', '&lt;'),
  money: v => Number(v).toFixed(4),
  plural: (n, word) => `${n} ${word}`,
  setMetrics() {},
};
vm.createContext(context);
const functions = source.slice(source.indexOf('function recordCall('), source.lastIndexOf('\ninit();'));
vm.runInContext('let recordedCalls = new Map(); let trainingUsd=0, jobs=0, llmUsd=0;\n' + functions, context);
vm.runInContext(`
  recordCall({id:'a', model:'<model>', tokens_in:100, tokens_out:20, cost_usd:0.01, attempt:1});
  recordCall({id:'b', model:'<model>', tokens_in:110, tokens_out:30, cost_usd:0.02, attempt:2});
  recordCall({id:'a', model:'<model>', tokens_in:100, tokens_out:20, cost_usd:0.01, attempt:1});
`, context);
let result = vm.runInContext('recordedCost()', context);
assert.equal(result.llm_calls['<model>'].calls, 2);
assert.equal(result.llm_calls['<model>'].tokens_in, 210);
assert.equal(result.llm_calls['<model>'].tokens_out, 50);
assert.equal(result.llm_cost_usd, 0.03);
assert.equal(nodes.costCard.hidden, false);
assert(nodes.costbody.innerHTML.includes('&lt;model>'));
assert(!nodes.costbody.innerHTML.includes('<model>'));
vm.runInContext('trainingUsd=0.2; jobs=4; showCost({llm_cost_usd:0.03});', context);
assert.equal(vm.runInContext('llmUsd', context), 0.03);
assert(nodes.costbody.innerHTML.includes('0.2300'));
// Old saved demos have no per-call events and must retain summary costs.
vm.runInContext('recordedCalls.clear(); trainingUsd=0; showCost({llm_cost_usd:0.4,llm_calls:{legacy:{calls:1,tokens_in:10,tokens_out:5,cost_usd:0.4}}});', context);
assert.equal(vm.runInContext('llmUsd', context), 0.4);
assert(nodes.costbody.innerHTML.includes('legacy'));
assert(!nodes.costbody.innerHTML.includes('Pending requests'));
// A fresh investigation starts a separate journal.
vm.runInContext("recordCall({id:'a',model:'new',tokens_in:1,tokens_out:1,cost_usd:0.005});", context);
assert.equal(vm.runInContext('llmUsd', context), 0.005);
"""
    subprocess.run(
        [node, "-e", script, str(page)], check=True, capture_output=True, text=True
    )
