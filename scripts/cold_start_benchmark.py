"""Measure cold vs warm invocation latency of the deployed function (evidence for the trade-off analysis).

A cold start is forced by changing an environment variable, which makes Lambda
discard existing execution environments. Results go to docs/evidence/cold_start.csv.

    python scripts/cold_start_benchmark.py --rounds 5 --warm 10
    AWS_ENDPOINT_URL=http://localhost:4566 python scripts/cold_start_benchmark.py   # LocalStack
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--function", default=os.getenv("LAMBDA_FUNCTION_NAME", "wine-inference-dev"))
    ap.add_argument("--rounds", type=int, default=5, help="number of forced cold starts")
    ap.add_argument("--warm", type=int, default=10, help="warm calls after each cold start")
    args = ap.parse_args()

    lam = boto3.client("lambda", endpoint_url=os.getenv("AWS_ENDPOINT_URL") or None)
    payload = (ROOT / "events" / "direct_invoke.json").read_bytes()
    rows = []
    for r in range(args.rounds):
        env = lam.get_function_configuration(FunctionName=args.function).get("Environment", {}).get("Variables", {})
        env["BENCHMARK_NONCE"] = str(time.time())  # forces fresh execution environments
        lam.update_function_configuration(FunctionName=args.function, Environment={"Variables": env})
        lam.get_waiter("function_updated_v2").wait(FunctionName=args.function)
        for i in range(args.warm + 1):
            t0 = time.perf_counter()
            resp = lam.invoke(FunctionName=args.function, Payload=payload)
            client_ms = (time.perf_counter() - t0) * 1000
            body = json.loads(json.loads(resp["Payload"].read())["body"])
            rows.append({"round": r, "call": i, "cold_start": body["cold_start"],
                         "client_ms": round(client_ms, 1), "handler_ms": body["latency_ms"]})

    out = ROOT / "docs" / "evidence" / "cold_start.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for label, flag in (("cold", True), ("warm", False)):
        vals = [r["client_ms"] for r in rows if r["cold_start"] is flag]
        if vals:
            print(f"{label:>4}: n={len(vals):3d}  median={statistics.median(vals):7.1f} ms  max={max(vals):7.1f} ms")
    print(f"raw data: {out}")


if __name__ == "__main__":
    main()
