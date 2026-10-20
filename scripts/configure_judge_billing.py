"""Import a locally saved billing request without executing captured shell code."""

import argparse
import json
import os
import shlex
from pathlib import Path

from apps.judge.balance import BALANCE_PATH, ORIGIN


def parse_capture(text):
    args = shlex.split(text.replace("\\\n", " "))
    if not args or args.pop(0) != "curl":
        raise ValueError("Expected Copy as cURL (bash)")
    headers, urls, body = {}, [], None
    while args:
        arg = args.pop(0)
        if arg in (
            "-H",
            "--header",
            "-b",
            "--cookie",
            "-d",
            "--data",
            "--data-raw",
            "--data-binary",
            "-X",
            "--request",
            "--url",
        ):
            if not args:
                raise ValueError("Incomplete captured request")
            value = args.pop(0)
            if arg in ("-H", "--header"):
                name, separator, content = value.partition(":")
                if not separator or "\n" in content or "\r" in content:
                    raise ValueError("Invalid captured header")
                headers[name.lower()] = content.strip()
            elif arg in ("-b", "--cookie"):
                if value.startswith("@"):
                    raise ValueError("Cookie file references are not accepted")
                headers["cookie"] = value
            elif arg in ("-d", "--data", "--data-raw", "--data-binary"):
                body = json.loads(value)
            elif arg in ("-X", "--request"):
                if value != "POST":
                    raise ValueError("Expected the balance POST request")
            else:
                urls.append(value)
        elif arg in ("--compressed", "--globoff"):
            continue
        elif arg.startswith("https://"):
            urls.append(arg)
        else:
            raise ValueError(
                "Unsupported captured option; copy the balance request as cURL"
            )
    if urls != [ORIGIN + BALANCE_PATH]:
        raise ValueError("Expected the Token Factory getBalance endpoint")
    if not isinstance(body, dict) or set(body) != {"contractId"}:
        raise ValueError("Expected a single billing contract")
    if not isinstance(body["contractId"], str) or not body["contractId"]:
        raise ValueError("Missing billing contract")
    if not headers.get("cookie") or not headers.get("x-csrf-token"):
        raise ValueError("The captured request has no billing session")
    return {"contract_id": body["contractId"], "headers": headers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = parse_capture(args.capture.read_text())
    except (ValueError, OSError):
        # shlex/JSON errors can quote secrets from the input.
        raise SystemExit(
            "Could not import the billing request. No command was executed."
        ) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(config, stream)
    args.out.chmod(0o600)
    print("Billing session saved privately. No command was executed.")


if __name__ == "__main__":
    main()
