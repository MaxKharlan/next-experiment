const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.match(script, /\ninit\(\);\s*$/);
const nodes = new Map();
const makeNode = () => ({
  textContent: '', innerHTML: '', hidden: false, disabled: false, value: '',
  parentElement: {hidden: false}, nextElementSibling: {}, style: {},
  classList: {remove() {}, add() {}}, removeAttribute(key) {delete this[key];},
  appendChild(node) {this.innerHTML += node.innerHTML;}, scrollIntoView() {},
});
const get = id => {if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id);};
const timers = new Map();
let nextTimer = 0;
const context = vm.createContext({
  document: {getElementById: get, documentElement: {dataset: {mode: 'static'}},
    body: makeNode(), createElement: makeNode},
  addEventListener() {}, matchMedia: () => ({matches: true}),
  setTimeout: fn => {timers.set(++nextTimer, fn); return nextTimer;},
  clearTimeout: id => timers.delete(id), console, URL, URLSearchParams,
});
vm.runInContext(script.replace(/\ninit\(\);\s*$/, ''), context);
const run = code => vm.runInContext(code, context);
get('demo').value = 'first';
run('staticMode([{name:"first"},{name:"second"}]);');
run('proof = {}; finish({});');
assert.equal(get('statusBadge').textContent, 'confirmed');
get('demo').value = 'second';
get('demo').onchange();
assert.equal(get('statusBadge').textContent, 'not started');
assert.equal(get('statusTitle').textContent, 'Ready to replay');
assert.equal(get('statusProof').textContent, '');
assert.equal(get('feed').innerHTML, '');
assert.equal(get('costCard').hidden, true);
assert.equal(get('reportActions').hidden, true);
assert.equal(run('proof'), null);
assert.equal(run('current'), null);
assert.match(get('about').innerHTML, /demo\/second.json/);
assert.equal(get('start').disabled, false);
const deferred = () => {let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject};};
const event = title => ({type: 'test', data: title, after_s: 0});
run('HANDLERS.test = title => setStatus(null, title, "running", "running");');
async function main() {
  // A cancelled HTTP response must not overwrite the newly selected case.
  let pending = deferred();
  context.fetch = () => pending.promise;
  let old = run('play("first")');
  get('demo').onchange();
  context.fetch = async () => ({ok:true, json:async () => [event('second loaded')]});
  await run('play("second")');
  pending.resolve({ok:true, json:async () => [event('stale first')]});
  await old;
  assert.equal(get('statusTitle').textContent, 'second loaded');
  // Selection can also change while the response body is being decoded.
  pending = deferred();
  context.fetch = async () => ({ok:true, json:() => pending.promise});
  old = run('play("first")');
  await Promise.resolve();
  get('demo').onchange();
  pending.resolve([event('stale decoded body')]);
  await old;
  assert.equal(get('statusBadge').textContent, 'not started');
  // Errors from cancelled loads must not mark the selected case as failed.
  pending = deferred();
  context.fetch = () => pending.promise;
  old = run('play("first")');
  get('demo').onchange();
  pending.reject(new Error('old request failed'));
  await old;
  assert.equal(get('statusBadge').textContent, 'not started');
  // Switching during timed playback clears the queued continuation.
  context.fetch = async () => ({ok:true, json:async () => [event('first'), event('late')]});
  await run('play("first")');
  assert.equal(timers.size, 1);
  get('demo').onchange();
  assert.equal(timers.size, 0);
  assert.equal(run('queue.length'), 0);
  // Malformed responses fail visibly and allow a retry.
  context.fetch = async () => ({ok:true, json:async () => {throw new Error('invalid JSON');}});
  await run('play("first")');
  assert.equal(get('statusBadge').textContent, 'failed');
  assert.equal(get('start').disabled, false);
  // Real measured results never acquire an agent confirmation or held-out score.
  const measured = JSON.parse(fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[2]), '../../../demo/answer-cut-tf.json'), 'utf8'));
  context.measured = measured;
  run('reset(); showStep(measured.report.steps[0]); finish({measured_experiment: measured.report.measured_experiment});');
  assert.equal(get('statusBadge').textContent, 'measured result');
  assert.equal(run('proof'), null);
  assert.equal(get('mHeldout').textContent, '');
  assert.match(get('feed').innerHTML, /92.6%/);
  assert.match(get('feed').innerHTML, /Permutation gate skipped/);
  assert.match(get('feed').innerHTML, /not an end-to-end agent investigation/);
  assert.match(get('feed').innerHTML, /ftjob-7cdfb67f667348c2aa8939a65d8f0be9/);
  // Nested raw evidence remains readable and HTML is escaped.
  run('reset(); showStep({kind:"probe", probe:"sample", outcome:"found", gain_bits:0, probs:{}, raw:{files:[{name:"<img src=x onerror=alert(1)>", count:2}]}});');
  assert.match(get('feed').innerHTML, /Raw evidence/);
  assert.match(get('feed').innerHTML, /&quot;files&quot;/);
  assert.match(get('feed').innerHTML, /&lt;img/);
  assert.doesNotMatch(get('feed').innerHTML, /\[object Object\]|<img/);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
