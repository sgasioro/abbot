"""Simulated thin-film anneal rig behind a local HTTP endpoint.

The agent under test never sees this file: it only gets ``measure.py``, a thin
client that posts one run here and prints the film's measurements. ``run.py``
starts and stops the rig around each agent turn.

The rig, as a person would describe it:
  anneal temperature 150-450 C, anneal time 10 s to 1 hour, stage position
  0-50 mm (the sample has a thickness gradient along it), substrate Si, sapphire
  or glass. The stage moves at most 2 mm between runs and starts at 10 mm. Each run
  gives conductivity and roughness; roughness has to stay under 3 nm.

What the person doesn't say:
  the furnace trips above 400 C (no measurement); the best film is near 330 C,
  about 630 s, 18 mm, on sapphire; conductivity noise is 5% of the signal plus a
  small floor, so a reading near zero can come out slightly negative.

The stage position is kept in ``--state`` so it survives restarts between turns.

    python benchmarks/agent/rig.py --port 8898 --state /path/to/stage.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

START_POSITION = 10.0
MAX_MOVE = 2.0
TRIP_TEMP = 400.0
SUBSTRATES = {"Si": 0.85, "sapphire": 1.0, "glass": 0.6}


def film(temp: float, time: float, position: float, substrate: str, rng) -> dict:
    """One anneal's measurements, with noise."""
    log_time = math.log10(max(time, 1e-3))
    signal = (
        1200.0
        * math.exp(-(((temp - 330.0) / 45.0) ** 2))
        * math.exp(-(((log_time - 2.8) / 0.5) ** 2))
        * math.exp(-(((position - 18.0) / 12.0) ** 2))
        * SUBSTRATES[substrate]
    )
    conductivity = signal * (1.0 + rng.gauss(0.0, 0.05)) + rng.gauss(0.0, 2.0)
    roughness = (
        0.8
        + 0.012 * max(0.0, temp - 250.0)
        + 0.9 * max(0.0, log_time - 2.5)
        + rng.gauss(0.0, 0.05)
    )
    return {
        "conductivity_S_per_cm": round(conductivity, 1),
        "roughness_nm": round(roughness, 3),
    }


def make_handler(state_path: Path, rng):
    def position() -> float:
        if state_path.exists():
            return json.loads(state_path.read_text())["position"]
        return START_POSITION

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            req = json.loads(self.rfile.read(length))
            temp, time = req["anneal_temp"], req["time"]
            x, substrate = req["position"], req["substrate"]
            if substrate not in SUBSTRATES:
                return self._send(400, {"error": f"unknown substrate {substrate!r}"})
            if not 0.0 <= x <= 50.0:
                return self._send(400, {"error": f"stage position {x} mm off the stage"})
            here = position()
            if abs(x - here) > MAX_MOVE + 1e-9:
                return self._send(409, {
                    "error": f"stage move from {here:.2f} mm to {x:.2f} mm exceeds "
                    f"{MAX_MOVE:g} mm; no measurement"
                })
            state_path.write_text(json.dumps({"position": x}))
            if temp > TRIP_TEMP:
                return self._send(500, {
                    "error": f"furnace over-temperature trip above {TRIP_TEMP:g} C; "
                    "no measurement"
                })
            self._send(200, film(temp, time, x, substrate, rng))

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    handler = make_handler(args.state, random.Random(args.seed))
    HTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
