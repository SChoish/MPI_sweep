#!/usr/bin/env python3
"""Preview or execute the predeclared K=3 recenter/fixed_ref training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("RECENTER_FIXED_REF_K3_PROTOCOL.json")
VERIFY = Path(__file__).with_name("verify_recenter_fixed_ref_k3.py")


def pilot_fingerprint(python: str, data: Path) -> dict:
    sources = (ROOT / "train_td3bc.py", PROTOCOL, VERIFY, Path(__file__))
    return {
        "python": str(Path(python).resolve()), "data_dir": str(data),
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sources},
    }


def choices(requested: list, available: list, label: str) -> list:
    invalid = set(requested) - set(available)
    if invalid:
        raise ValueError(f"{label} not in frozen protocol: {sorted(invalid)}")
    return requested or available


def training_command(python: str, output: Path, data: Path, env: str,
                     tau: float, seed: int, branch: str, pilot: bool) -> list[str]:
    step = 1024 if pilot else 1_000_000
    return [
        python, str(ROOT / "train_td3bc.py"), "--method", "shared",
        "--deployment-branch", branch, "--integrator", "implicit",
        "--mpi-steps", "3", "--env", env, "--tau", f"{tau:g}",
        "--seed", str(seed), "--max-timesteps", str(step),
        "--eval-freq", str(step if pilot else 5000),
        "--eval-episodes", "10", "--save-interval", str(step if pilot else 100000),
        "--data-dir", str(data), "--save-dir", str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually start training")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--tau", type=float, action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.pilot and (args.env or args.tau or args.seed):
        parser.error("--pilot uses the fixed pilot cell; do not override its grid")
    if args.pilot:
        pilot = protocol["pilot"]
        envs, taus, seeds = [pilot["environment"]], [pilot["T"]], [pilot["seed"]]
    else:
        envs = choices(args.env, protocol["environments"], "environment")
        taus = choices(args.tau, protocol["total_horizon_T"], "T")
        seeds = choices(args.seed, protocol["seeds"], "seed")
    pairs = [(env, tau, seed) for env in envs for tau in taus for seed in seeds]
    if len(pairs) != len(set(pairs)):
        parser.error("duplicate environment/T/seed selection")
    save_root = args.save_dir.resolve()
    output = save_root / ("pilot" if args.pilot else "full")
    data = args.data_dir.resolve()
    step = 1024 if args.pilot else 1_000_000
    print(json.dumps({"mode": "execute" if args.execute else "preview",
                      "pairs": len(pairs), "training_runs": 2 * len(pairs),
                      "steps_per_run": step, "output": str(output)}), flush=True)
    if not args.execute:
        env, tau, seed = pairs[0]
        for branch in ("recenter", "fixed_ref"):
            print("example: " + shlex.join(training_command(
                args.python, output, data, env, tau, seed, branch, args.pilot
            )))
        return

    marker = save_root / "PILOT_VERIFIED.json"
    fingerprint = pilot_fingerprint(args.python, data)
    if not args.pilot:
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
            parser.error("run the verified pilot with this interpreter and code before full training")

    for index, (env, tau, seed) in enumerate(pairs, start=1):
        print(f"[pair {index}/{len(pairs)}] {env} T={tau:g} seed={seed}", flush=True)
        for branch in ("recenter", "fixed_ref"):
            command = training_command(args.python, output, data, env, tau,
                                       seed, branch, args.pilot)
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        verify_cmd = [args.python, str(VERIFY), "--save-dir", str(output),
                      "--env", env, "--tau", f"{tau:g}", "--seed", str(seed),
                      "--step", str(step)]
        subprocess.run(verify_cmd, cwd=ROOT, check=True)
    if args.pilot:
        marker.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
        print(f"[pilot] verified gate written: {marker}", flush=True)


if __name__ == "__main__":
    main()
