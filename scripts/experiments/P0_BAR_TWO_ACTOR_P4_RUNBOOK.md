# P0 BAR-P4 versus two-actor-P4 runbook

## Release status

This repository publishes the experiment code and verification code only. It
contains no frozen P0 manifest, launch registry, job output, checkpoint, or P0
result. A result is admissible only after a clean operator follows this runbook
and the independent verifier passes.

The experiment is one contemporaneous, paired grid:

- nine D4RL task variants;
- `T = {4, 7, 10, 14, 20}`;
- seeds `0` and `1` only;
- 90 BAR-P4 runs plus 90 two-actor-P4 runs, for 180 launches total.

The nine task variants are exactly:

- `hopper-medium-v2`;
- `hopper-medium-replay-v2`;
- `hopper-expert-v2`;
- `halfcheetah-medium-v2`;
- `halfcheetah-medium-replay-v2`;
- `halfcheetah-expert-v2`;
- `walker2d-medium-v2`;
- `walker2d-medium-replay-v2`;
- `walker2d-expert-v2`.

Do not add seeds 4 through 7. Do not splice a historical P4 score, checkpoint,
configuration, or evaluation file into either arm. Both arms must be generated
from the same frozen manifest. The internal compatibility arm is the
two-actor policy-separation control (legacy `mcep` token; not published
reproduction).

After `freeze`, `FROZEN_MANIFEST.json` and its SHA-256 sidecar are the single
source of truth for the run. This runbook and `TODO.md` state intent, but they do
not override the frozen code, configuration, dependency, dataset,
normalization, hardware, evaluation, output, analysis, or per-run contracts in
the manifest. Never edit the manifest or regenerate its sidecar.

## Paths and preflight

All mutable artifacts must live under an absolute external root, outside the
source checkout. Set these values for the host that will run the experiment:

```bash
export P0_PYTHON=/absolute/path/to/the/frozen/environment/bin/python
export P0_DATA_ROOT=/absolute/external/path/to/d4rl-data
export P0_ROOT=/absolute/external/path/to/p0-bar-p4-two-actor-p4
export P0_RUNS="$P0_ROOT/runs"
export P0_MANIFEST="$P0_ROOT/contracts/FROZEN_MANIFEST.json"
export P0_CACHE="$P0_ROOT/xla-cache"
export P0_ANALYSIS="$P0_ROOT/analysis-v1"
```

Run from the repository root. Before freezing, `git status` must print no
entries, the desired revision must already be committed, all nine dataset files
must be present, the selected Python environment must be final, and the GPU
inventory must be stable:

```bash
git status --short --untracked-files=all
git rev-parse HEAD
"$P0_PYTHON" --version
nvidia-smi -L
```

`freeze` rejects a dirty checkout, a different `--python` from the interpreter
running the command, duplicate or unavailable GPU IDs, missing datasets, or an
output root inside the checkout. It computes dataset and normalization hashes
and records the exact installed-package and GPU inventories.

## 1. Plan without launching

Invoking the tool without a subcommand defaults to `plan`. The explicit form
below prints the fixed grid
and writes nothing:

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py plan \
  --output-root "$P0_RUNS"
```

Confirm `total_runs` is 180 and each condition is 90.

## 2. Freeze the single source of truth

This is create-only: neither the manifest nor its `.sha256` sidecar may already
exist.

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py freeze \
  --manifest "$P0_MANIFEST" \
  --data-dir "$P0_DATA_ROOT" \
  --output-root "$P0_RUNS" \
  --python "$P0_PYTHON" \
  --gpus 0,1 \
  --slots-per-gpu 1 \
  --compilation-cache-dir "$P0_CACHE"
```

Change `--gpus` and `--slots-per-gpu` only before this command, based on the
frozen hardware. Do not hand-edit those values afterward.

## 3. Preview the launch

`launch` remains read-only unless `--execute` is present. It rechecks the
manifest sidecar and the frozen source, dependencies, hardware, datasets, and
normalization before printing all pending commands:

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py launch \
  --manifest "$P0_MANIFEST"
```

Review the 180 fresh commands and their paired BAR/control ordering. A preview
must show `execute=False`.

The prelaunch scan is deliberately strict. A run directory counts as complete
only when its config, exact eval schema and final row, and final checkpoint all
validate. Otherwise, the highest numeric `params_<step>.pkl` with
`0 < step < 1000000` is the only resume candidate, and it must have an exact
entry in the supplied sealed registry. Any other stale or nonempty run
directory hard-fails the whole preview or launch; it is never silently skipped,
reused, or overwritten.

## 4. Launch explicitly

Only this explicit flag starts jobs:

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py launch \
  --manifest "$P0_MANIFEST" \
  --execute
```

The harness places run directories under `$P0_RUNS/bar_p4` and
`$P0_RUNS/two_actor_p4`, and logs under `$P0_RUNS/logs`. It never uses an
unsealed partial checkpoint automatically.

## 5. Seal and preview a resume

After an interruption, identify the latest trusted partial checkpoint for each
interrupted run. Here, "trusted" means an operator has established that the
local pickle was produced by the corresponding run under this frozen manifest;
pickle provenance cannot be inferred from its contents. The harness selects
the greatest numeric step suffix in that run directory, requires the payload
step to equal the filename step, and requires `0 < step < 1000000`. Seal that
exact latest file for every interrupted run, repeating `--checkpoint`; do not
seal an older checkpoint while a later partial exists. The registry and
sidecar are create-only and tied to the frozen manifest hash:

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py seal-resume \
  --manifest "$P0_MANIFEST" \
  --output "$P0_ROOT/contracts/RESUME_REGISTRY_v1.json" \
  --checkpoint "$P0_RUNS/bar_p4/BAR_RUN_TAG/params_250000.pkl" \
  --checkpoint "$P0_RUNS/two_actor_p4/TWO_ACTOR_RUN_TAG/params_250000.pkl"
```

Use the actual tags and steps. The seal validates the checkpoint step, exact
payload keys, scientific configuration, actor/state count, dataset
normalization, finite parameter and optimizer leaves, and hashes of the file
and weights. It also requires `critic_step == step` and every actor step to
equal `step // 2`, because the frozen `policy_freq` is 2; at the final
checkpoint these are 1,000,000 and 500,000. Preview again before executing:

```bash
"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py launch \
  --manifest "$P0_MANIFEST" \
  --resume-registry "$P0_ROOT/contracts/RESUME_REGISTRY_v1.json"

"$P0_PYTHON" scripts/experiments/run_p0_bar_two_actor_p4.py launch \
  --manifest "$P0_MANIFEST" \
  --resume-registry "$P0_ROOT/contracts/RESUME_REGISTRY_v1.json" \
  --execute
```

If another interruption changes the set of latest partial checkpoints, create
a new complete registry at a new path; never overwrite a prior registry.

## 6. Analyze all final artifacts

Analysis requires one exact final row and `params_1000000.pkl` for every one of
the 180 runs. `$P0_ANALYSIS` must be absent or empty. Before reading results,
the analyzer rechecks the frozen Git revision, relevant source-file hashes,
Python binary, resolved packages, and requirements hash. It then validates final
checkpoint payloads and writes create-only paired scores, summary,
final-checkpoint fingerprints, and an analysis manifest:

```bash
"$P0_PYTHON" scripts/diagnostics/analyze_p0_bar_two_actor_p4.py \
  --manifest "$P0_MANIFEST" \
  --output-dir "$P0_ANALYSIS"
```

The deployment-score contract is:

| Procedure | Target branch, retained for diagnosis | Deployment score used in P0 |
| --- | --- | --- |
| BAR-P4 | actor 1: `return`, `d4rl_score` | actor 4: `return_pi4`, `d4rl_pi4` |
| two-actor-P4 | target actor: `return`, `d4rl_score` | deployment actor: `return_eval`, `d4rl_eval` |

The primary continuous contrast is `d4rl_pi4 - d4rl_eval`, summarized as the
mean of nine fixed task-variant means. The report also includes the paired
median, wins/ties, task means, a 100,000-draw task-variant bootstrap interval,
and collapse transitions at strict thresholds 0, 10, 20, 30, and 40.

The prelocked decision labels apply to the full four-actor procedure versus the
full two-actor procedure:

- `four_actor_procedure_supporting`: interval lower bound is above 0 and the
  point estimate is at least +3;
- `two_actor_procedure_supporting`: interval upper bound is below 0 and the
  point estimate is at most -3;
- `author_band_comparable`: the full interval lies within [-3, +3];
- `unresolved`: every other outcome.

The rules use the declared order above as precedence: the first satisfied label
wins. This matters at the exact +3 or -3 boundary, where a directional rule can
also fit inside the author band; the directional procedure-supporting label is
applied first.

The 3-point minimum worthwhile difference is author-defined, not an externally
validated equivalence margin. The continuous contrast is primary; collapse
summaries cannot override its label.

## 7. Verify independently

The verifier does not import the analyzer or launch harness. It independently
rechecks the frozen Git revision, relevant source-file hashes, Python binary,
resolved packages, and requirements hash before reloading the trusted local
final checkpoints. It then reconstructs all 180 keys, 90 pairs, raw hashes,
compact checkpoint fingerprints, statistics, bootstrap, collapse tables, and
decision gate:

```bash
"$P0_PYTHON" scripts/diagnostics/verify_p0_bar_two_actor_p4.py \
  --manifest "$P0_MANIFEST" \
  --analysis-dir "$P0_ANALYSIS"
```

On success, the command creates `$P0_ANALYSIS/VERIFY.json` and
`$P0_ANALYSIS/VERIFY.json.sha256`. The create-only receipt binds the frozen
manifest, analysis manifest, raw artifact tree, final-checkpoint tree, verifier
source, and reported full-procedure outcome. Neither file may already exist. To
repeat verification later, provide a new receipt path with `--receipt`; never
overwrite an earlier receipt.

Publish or interpret a P0 result only after this command prints `PASS` and the
receipt plus sidecar have been retained.

## Scope limit

The resampling unit is the fixed set of nine task variants. Those variants
share only three dynamics families (Hopper, HalfCheetah, and Walker2d) and one
benchmark suite, so they are not nine broadly independent environments. The
result is scoped to this exact 9-task x 5-T x 2-seed grid and does not establish
broader offline-RL generalization or isolate re-centering as a causal factor.
The verifier establishes consistency of the supplied artifacts with the frozen
contract; it does not cryptographically establish who launched a job or when.
Retain the external scheduler/log provenance, and enforce the ban on historical
P4 splicing as an operator provenance requirement.
