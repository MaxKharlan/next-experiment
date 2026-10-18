import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main


@pytest.mark.parametrize("public", [False, True])
def test_capabilities_hide_paid_form_in_public_demo(monkeypatch, public):
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=public)
    )
    with TestClient(main.app) as client:
        response = client.get("/api/capabilities")
    assert response.status_code == 200
    assert response.json() == {"explicit_cases": not public}


def test_case_form_requires_validation_and_one_explicit_start():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to test browser handlers")
    page = Path(__file__).resolve().parents[2] / "apps/api/static/index.html"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const text=fs.readFileSync(process.argv[1], 'utf8');
const nodes={}; const el=id=>nodes[id] ||= {value:'',disabled:false,textContent:''};
const requests=[]; const watched=[];
const context={ $:el, crypto:{randomUUID:()=> 'test-request-key'}, money:x=>String(x), history:{replaceState(){}}, watch:id=>watched.push(id),
  fetch:async (url, options)=>{
    requests.push([url,JSON.parse(options.body)]);
    if(url.endsWith('/from-case')) assert.equal(options.headers['Idempotency-Key'],'test-request-key');
    if(url.endsWith('/validate')) return {ok:true,json:async()=>({protocol:{job_id:'job',model:'model',budget_usd:1,pair_estimate_usd:0.05,datasets:{train:{examples:10}}}})};
    return {ok:true,json:async()=>({id:'inv-test'})};
  }
};
vm.createContext(context);
vm.runInContext(text.slice(text.indexOf('let validatedCaseText'),text.indexOf('function showMethod()')),context);
(async()=>{
  await el('launchCase').onclick(); assert.equal(requests.length,0);
  el('caseRequest').value='invalid'; await el('validateCase').onclick();
  assert.equal(requests.length,0); assert.equal(el('launchCase').disabled,true);
  el('caseRequest').value='{"case":{},"experiment":{}}';
  await el('validateCase').onclick();
  assert.equal(requests.length,1); assert(requests[0][0].endsWith('/validate'));
  assert.equal(el('launchCase').disabled,false);
  assert(el('caseFeedback').textContent.includes('No training started'));
  el('caseRequest').value='{}'; el('caseRequest').oninput();
  await el('launchCase').onclick(); assert.equal(requests.length,1);
  el('caseRequest').value='{"case":{},"experiment":{}}';
  await el('validateCase').onclick();
  await Promise.all([el('launchCase').onclick(),el('launchCase').onclick()]);
  assert.equal(requests.filter(([url])=>url.endsWith('/from-case')).length,1);
  assert.deepEqual(watched,['inv-test']);
  assert.equal(el('launchCase').disabled,true);
  // Server errors keep launch unavailable and are rendered as text.
  context.fetch=async()=>({ok:false,json:async()=>({detail:'<bad dataset>'})});
  await el('validateCase').onclick();
  assert.equal(el('caseFeedback').textContent,'<bad dataset>');
  assert.equal(el('launchCase').disabled,true);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    subprocess.run(
        [node, "-e", script, str(page)], check=True, capture_output=True, text=True
    )
