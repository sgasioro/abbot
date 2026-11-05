"""Client for the anneal rig: sets up one anneal and prints the film measurement.

Usage: python3 measure.py --anneal-temp C --time SECONDS --position MM --substrate NAME
Prints JSON: {"conductivity_S_per_cm": ..., "roughness_nm": ...}. Exits 1 on a rig error.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

RIG = os.environ.get("ANNEAL_RIG_URL", "http://127.0.0.1:8898/run")

parser = argparse.ArgumentParser(description="Run one anneal on the rig and measure the film.")
parser.add_argument("--anneal-temp", type=float, required=True, help="anneal temperature, C")
parser.add_argument("--time", type=float, required=True, help="anneal time, seconds")
parser.add_argument("--position", type=float, required=True, help="stage position, mm")
parser.add_argument("--substrate", required=True, help="substrate loaded (Si, sapphire or glass)")
args = parser.parse_args()
body = json.dumps({
    "anneal_temp": args.anneal_temp, "time": args.time,
    "position": args.position, "substrate": args.substrate,
}).encode()
request = urllib.request.Request(RIG, data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode())
except urllib.error.HTTPError as err:
    print("RIG ERROR: " + json.loads(err.read().decode())["error"], file=sys.stderr)
    sys.exit(1)
