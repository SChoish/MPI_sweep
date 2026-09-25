#!/usr/bin/env python3
"""Preview or execute the predeclared K=4 recenter/fixed_ref training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("RECENTER_FIXED_REF_K4_PROTOCOL.json")
VERIFY = Path(__file__).with_name("verify_recenter_fixed_ref_k4.py")


def pilot_fingerprint(python: str, data: Path, host: str) -> dict:
    sources = (ROOT / "train_td3bc.py", PROTOCOL, VERIFY, Path(__file__))
    return {
        "python": str(Path(python).resolve()), "data_dir": str(data), "host": host,
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sources},
    }


def training_command(python: str, output: Path, data: Path, env: str,
                     tau: float, seed: int, branch: str, pilot: bool) -> list[str]:
    step = 1024 if pilot else 1_000_000
    return [
        python, str(ROOT / "train_td3bc.py"), "--method", "shared",
        "--deployment-branch", branch, "--integrator", "implicit",
        "--mpi-steps", "4", "--env", env, "--tau", f"{tau:g}",
        "--seed", str(seed), "--max-timesteps", str(step),
        "--eval-freq", str(step),
        "--eval-episodes", "10", "--save-interval", str(step if pilot else 100000),
        "--data-dir", str(data), "--save-dir", str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--host", choices=("svcho", "choi"), required=True)
    parser.add_argument("--stage", choices=("primary", "boundary"), default="primary")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually start training")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.pilot:
        pilot = protocol["pilot"]
        pairs = [(pilot["environment"], pilot["T"], pilot["seed"])]
    else:
        host_parity = 0 if args.host == "svcho" else 1
        pairs = [
            (env, tau, seed)
            for env_index, env in enumerate(protocol["environments"])
            for tau in protocol["stages"][args.stage]
            for seed in protocol["seeds"]
            if (env_index + seed) % 2 == host_parity
        ]
    if len(pairs) != len(set(pairs)):
        parser.error("duplicate environment/T/seed selection")
    save_root = args.save_dir.resolve()
    output = save_root / ("pilot" if args.pilot else "full")
    data = args.data_dir.resolve()
    step = 1024 if args.pilot else 1_000_000
    print(json.dumps({"mode": "execute" if args.execute else "preview",
                      "host": args.host, "stage": "pilot" if args.pilot else args.stage,
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
    fingerprint = pilot_fingerprint(args.python, data, args.host)
    if not args.pilot:
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
            parser.error("run the verified pilot with this interpreter and code before full training")

    for index, (env, tau, seed) in enumerate(pairs, start=1):
        print(f"[pair {index}/{len(pairs)}] {env} T={tau:g} seed={seed}", flush=True)
        verify_cmd = [args.python, str(VERIFY), "--save-dir", str(output),
                      "--env", env, "--tau", f"{tau:g}", "--seed", str(seed),
                      "--step", str(step)]
        if not args.pilot and all(
            (output / f"{env}_tau{tau:g}_shared_{branch}4_seed{seed}"
             / f"params_{step}.pkl").is_file()
            for branch in ("recenter", "fixed_ref")
        ):
            subprocess.run(verify_cmd, cwd=ROOT, check=True)
            print("[skip] final pair already verified", flush=True)
            continue
        for branch in ("recenter", "fixed_ref"):
            command = training_command(args.python, output, data, env, tau,
                                       seed, branch, args.pilot)
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(verify_cmd, cwd=ROOT, check=True)
    if args.pilot:
        marker.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
        print(f"[pilot] verified gate written: {marker}", flush=True)


if __name__ == "__main__":
    main()
