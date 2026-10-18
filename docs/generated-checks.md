# B4: generated checks through the local Docker runner

`investigate(..., sandbox_check={"question": ..., "data": {...}})` explicitly
opts a trusted Python caller into one generated check before hypothesis generation.
Super generates a bounded list of Python source lines through the existing
`probe_code` router role and versioned prompt. Pydantic validates Python syntax;
failed structured attempts use the connector's existing logged retries. Host-side
syntax parsing does not execute the source.

Only the supplied JSON data is bound to `DATA` in the container. Neither the
investigation context nor provider keys are forwarded. Select nonsensitive input
explicitly: the supplied data goes to Super, and generated code can echo it in
its output. Data embedded in source is visible in process/container metadata.
The container uses the restrictions documented in [Docker checks](docker-checks.md),
with a ten-second deadline and 8 KiB per output stream. No host execution fallback
exists. Provision the trusted Python image before enabling this feature.

The check must print one finite JSON object. Timeout, output overflow, nonzero
exit, unavailable Docker and malformed output produce distinct failure statuses.
A successful process alone is insufficient. The execution and code are stored
in a `sandbox_check` investigation step, with the input hash and prompt version;
the copied hypothesis context receives this record. No automatic causal verdict
or calibrated Bayesian update follows from a generated program's output.

Generated checks are disabled by default and rejected when `PUBLIC_DEMO=true`.
They are not exposed as a public REST option. The caller must have remaining
investigation budget; generation cost is logged by the router. This is not a
hard token-cost reservation system. Docker execution does not require provider
credits. The Token Factory Sandbox beta request was submitted on October 19, 2026;
Microsoft Forms displayed its submission confirmation. The optional email field
was left empty. Cloud beta activation is still pending. The saved
[confirmation](results/b467-20261019/sandbox/beta-request-confirmation.png) records
the submission. The beta request page is
[Token Factory Sandboxes](https://tokenfactory.nebius.com/sandboxes/about).

## Verification

```bash
RUN_DOCKER_SANDBOX_TESTS=1 uv run pytest -q \
  tests/unit/test_generated_check.py tests/unit/test_sandbox.py tests/unit/test_loop.py
```

A live Super call generated Python that Docker executed against three synthetic
answers: it reported `total=3` and `invalid_json=1`. The saved
[live result](results/b467-20261019/sandbox/live-check.json) contains generated
source, execution status and recorded usage ($0.0004005). Earlier attempts returned
incomplete code or HTML, and were rejected.
A [local replay](results/b467-20261019/sandbox/replay.json) also records an explicit
synthetic input, source hash and matching output without another model call.
The failed records are saved alongside the successful result. Another three-attempt
syntax-validation failure did not save its usage file; the experiment ledger
reserves $0.10 for that call instead of reporting an invented exact charge.

Tests also cover an absent host secret in the actual container, failure
classification, input limits, public-demo gating and forwarding results to the
hypothesis step. This verifies the local path, not resistance to kernel exploits
or availability of the cloud beta service.
