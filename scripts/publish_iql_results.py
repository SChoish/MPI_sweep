#!/usr/bin/env python3
"""Export completed IQL runs on their owning machine; never load training libraries.

With --push, publish from a disposable Git worktree so active training checkouts
and their local changes are untouched. --watch repeats every five minutes.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

FINAL_STEP = 1_000_000
PROFILES = {
    "ext_csv": ("/raid/ext_csv/MPI_store/iql_qbc_gaussian_w2_hc_walker_s0123",
                "iql_qbc_gaussian_w2_hc_walker_s0123", "qbc_gaussian_w2"),
    "ext_csh": ("/home/ext_csh/MPI_sweep/results/iql_awr_fr", "iql_awr_fr", "awr_gaussian_fr"),
}
FIELDS = ("env", "tau", "K", "seed", "variant", "geometry", "gaussian_qbc_mode",
          "policy", "hop", "eval_mode", "step", "status", "d4rl_score", "run",
          "machine", "hostname", "source", "schema", "eval_sha256")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature_hash(signature):
    return digest(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode())[:12]


def verified_run(directory, variant, machine):
    """Read final observations with COMPLETE/config/eval/checkpoint-size evidence.

    Checkpoints stay local and are not deserialized or rehashed on every tick.
    Their recorded checksum is retained in the manifest, not independently claimed.
    """
    config = json.loads((directory / "config.json").read_text())
    complete = json.loads((directory / "COMPLETE.json").read_text())
    signature = config["signature"]
    algorithm = signature["algorithm"]
    if signature["schema"] not in ("iql_actor_geometry_v5_consistent_gaussian",
                                    "iql_actor_geometry_v6_baseline_h"):
        raise ValueError("unsupported training schema")
    if (complete["schema"] != signature["schema"] or complete["step"] != FINAL_STEP
            or complete["signature"] != signature_hash(signature)):
        raise ValueError("completion/signature mismatch")
    data = (directory / "eval.csv").read_bytes()
    if complete["eval_sha256"] != digest(data):
        raise ValueError("evaluation hash mismatch")
    checkpoint = directory / f"params_{FINAL_STEP}.pkl"
    if checkpoint.stat().st_size != complete["checkpoint_bytes"] or complete["checkpoint_bytes"] <= 0:
        raise ValueError("final checkpoint size mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", complete["checkpoint_sha256"]):
        raise ValueError("checkpoint checksum missing from completion marker")
    k, seed, tau = int(algorithm["mpi_steps"]), int(signature["seed"]), float(algorithm["tau"])
    if k not in (1, 2, 3, 4) or seed not in range(4) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("run outside main experiment K/seed/T protocol")
    if variant not in algorithm["variants"]:
        raise ValueError("profile variant absent from run configuration")
    final = [r for r in csv.DictReader(io.StringIO(data.decode())) if int(r["step"]) == FINAL_STEP]
    policies = [("baseline", 1)]
    if k > 1:
        hops = range(1, k + 1) if signature["eval_hops"] == "all" else (1, k)
        policies.extend(("mpi", hop) for hop in hops)
    expected = set()
    for v in algorithm["variants"]:
        modes = ("mean",) if v == "qbc_deterministic_w2" else (
            ("mean", "sample") if signature["eval_mode"] == "both" else (signature["eval_mode"],))
        expected.update((v, policy, hop, mode) for policy, hop in policies for mode in modes)
    actual = [(r["variant"], r["policy"], int(r["hop"]), r["eval_mode"]) for r in final]
    if set(actual) != expected or len(actual) != len(expected) or complete["evaluation_rows"] != len(final):
        raise ValueError("missing or duplicate final evaluations")
    target = (variant, "baseline" if k == 1 else "mpi", k, "mean")
    if target not in expected:
        raise ValueError("final mean-action evaluation missing")
    row = final[actual.index(target)]
    geometry = "fr" if variant == "awr_gaussian_fr" else "w2"
    if row["refinement_geometry"] != geometry:
        raise ValueError("wrong refinement geometry")
    if variant == "qbc_gaussian_w2" and (
            algorithm["gaussian_qbc_mode"] != "stochastic" or row["gaussian_qbc_mode"] != "stochastic"):
        raise ValueError("Gaussian main experiment requires stochastic Q+BC")
    if int(row["K"]) != k or float(row["T"]) != tau:
        raise ValueError("evaluation K/T does not match configuration")
    if not math.isfinite(float(row["d4rl_score"])):
        raise ValueError("nonfinite final score")
    result = dict(env=signature["env"], tau=f"{tau:g}", K=k, seed=seed, variant=variant,
                  geometry=geometry, gaussian_qbc_mode=row.get("gaussian_qbc_mode", "na"),
                  policy=target[1], hop=k, eval_mode="mean", step=FINAL_STEP,
                  status="completed", d4rl_score=row["d4rl_score"], run=directory.name,
                  machine=machine, hostname=socket.gethostname(), source=str(directory),
                  schema=signature["schema"], eval_sha256=complete["eval_sha256"])
    evidence = dict(run=directory.name, signature=signature, sources=config.get("sources", {}),
                    completion=complete, checkpoint_size_checked=True, checkpoint_rehashed=False)
    return result, evidence


def export(source, variant, machine):
    if not source.is_dir():
        raise ValueError(f"source directory unavailable on this machine: {source}")
    rows, evidence, errors, excluded = [], [], [], []
    pending = 0
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        if not (directory / "COMPLETE.json").is_file():
            pending += 1
            continue
        try:
            # Historical pilots may share the source directory. Match the current queue.
            signature = json.loads((directory / "config.json").read_text())["signature"]
            algorithm = signature["algorithm"]
            domains = ("halfcheetah", "walker2d") if machine == "ext_csv" else ("hopper", "halfcheetah", "walker2d")
            envs = {f"{domain}-{dataset}-v2" for domain in domains
                    for dataset in ("medium", "medium-replay", "expert")}
            taus = (.05, .1, .2, .4, .7, 1.5, 2.5, 4., 7., 10.) if machine == "ext_csv" else (.05, .1, .4, 1.5, 2.5, 4., 10.)
            if (signature["env"] not in envs or algorithm["mpi_steps"] not in (1, 2, 3)
                    or float(algorithm["tau"]) not in taus or signature["seed"] not in range(4)):
                excluded.append(dict(run=directory.name, reason="outside_current_queue", signature=signature))
                continue
            row, proof = verified_run(directory, variant, machine)
            rows.append(row)
            evidence.append(proof)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{directory.name}: {error}")
    if errors:
        raise ValueError("Refusing to replace scores with an invalid snapshot:\n" + "\n".join(errors))
    if not rows:
        raise ValueError("No verified final scores; existing published files are untouched")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest = dict(schema="iql_final_score_export_v1", machine=machine, hostname=socket.gethostname(),
                    source=str(source), final_step=FINAL_STEP, complete_runs=len(rows),
                    pending_directories=pending, excluded=excluded, runs=evidence)
    return stream.getvalue(), json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", len(rows)


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def publish(repo, folder, contents, machine):
    remote = git(repo, "remote", "get-url", "origin").stdout.strip()
    if not re.search(r"[:/]SChoish/MPI_sweep(?:\.git)?/?$", remote):
        raise ValueError("publication requires origin SChoish/MPI_sweep")
    # Only these two paths are ever staged. The machine's existing STATUS is retained.
    paths = [f"sweep_results/{folder}/scores_verified.csv", f"sweep_results/{folder}/EXPORT.json"]
    git(repo, "fetch", "origin", "main")
    with tempfile.TemporaryDirectory(prefix="mpi-iql-publish-") as tmp:
        worktree = Path(tmp) / "checkout"
        git(repo, "worktree", "add", "--detach", str(worktree), "origin/main")
        try:
            for attempt in range(5):
                # Rebuild only our two generated files on the latest main on a push race.
                git(worktree, "pull", "--rebase", "origin", "main")
                for path, content in zip(paths, contents):
                    dest = worktree / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content)
                git(worktree, "add", "--", *paths)
                changed = git(worktree, "diff", "--cached", "--name-only").stdout.splitlines()
                if not changed:
                    print("Verified scores unchanged; no commit needed", flush=True)
                    return
                if set(changed) - set(paths):
                    raise ValueError("unexpected staged publication path")
                git(worktree, "commit", "-m", f"Publish verified {machine} IQL final scores")
                pushed = git(worktree, "push", "origin", "HEAD:main", check=False)
                if pushed.returncode == 0:
                    print("Published: " + git(worktree, "rev-parse", "HEAD").stdout.strip(), flush=True)
                    return
                # Discard only our disposable checkout's unpushed generated commit.
                git(worktree, "fetch", "origin", "main")
                git(worktree, "reset", "--hard", "origin/main")
            raise RuntimeError("publication failed after five attempts: " + pushed.stderr)
        finally:
            git(repo, "worktree", "remove", "--force", str(worktree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, help="override local raw run directory")
    parser.add_argument("--output", type=Path, help="local export folder; do not push")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--watch", action="store_true", help="repeat every five minutes; keeps running")
    args = parser.parse_args()
    if args.push == bool(args.output):
        parser.error("choose exactly one of --push or --output")
    default_source, folder, variant = PROFILES[args.profile]
    source = (args.source or Path(default_source)).resolve()
    # A watch process and a one-shot invocation cannot publish the same profile concurrently.
    lock_dir = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
    with (lock_dir / f"mpi-iql-publish-{os.getuid()}-{args.profile}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                scores, manifest, count = export(source, variant, args.profile)
                if args.push:
                    publish(args.repo.resolve(), folder, (scores, manifest), args.profile)
                else:
                    args.output.mkdir(parents=True, exist_ok=True)
                    for name, content in (("scores_verified.csv", scores), ("EXPORT.json", manifest)):
                        tmp = args.output / (name + ".tmp")
                        tmp.write_text(content)
                        tmp.replace(args.output / name)
                print(f"{args.profile}: {count} verified final scores", flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                detail = getattr(error, "stderr", "")
                print(f"Publication stopped for this tick: {error}\n{detail}", flush=True)
                if not args.watch:
                    raise SystemExit(1)
            if not args.watch:
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
