# Free testing

Open [Next Experiment](https://maxkharlan.github.io/next-experiment/#labelAudit).
No registration, API key or payment is needed. The public browser application
computes training-label checks on inputs you can edit or load from your device.
Its audit runs locally; inputs are not sent to a server.

1. Load **EOS accidentally masked** and click **Run audit in this browser**.
   The result reports one masked EOS target.
2. In the observed labels, replace the final `-100` with `99`. Rerun.
   Labels now align with the reference and EOS is preserved.
3. Try **Prompt included in the loss** and **Labels shifted twice**, or load
   your own JSON request. The page explains the input format and limits.
4. Download the audit and original input as JSON. Editing the input clears
   the previous result so it cannot be mistaken for a new check.
5. Select a saved investigation above and press **Replay** to inspect its
   provider calls, experiments and outcome. Download its original report.

The three label checks use the same logic as the Python application. Tests
compare the browser and Python results on more than 1,250 valid and invalid
requests. They inspect alignment, prompt supervision and EOS masking against
the reference supplied by the tester. They do not validate that reference's
meaning or establish a cause of model behavior.

Replay shows recorded results. Browser label audits compute new results.
Fresh Nemotron inference, Tavily searches and fine-tuning require the
self-hosted application and provider credentials; those paid operations are
not offered by the free browser test. The public site contains no credentials
and cannot submit paid work.

The application and saved reports are served by GitHub Pages without a
session expiry or per-user allowance. Keep the repository and Pages deployment
public through the judging period ending 15 December 2026.
