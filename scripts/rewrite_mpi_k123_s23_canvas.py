#!/usr/bin/env python3
"""Rewrite mpi-k123-s23.canvas.tsx from local MPI_sweep results (seeds 2,3)."""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_TAUS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0]
HI_TAUS = [24.0, 28.0, 34.0, 40.0]
TAUS = BASE_TAUS + HI_TAUS
ENVS = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
]
SHORT = {
    "hopper-medium-v2": "hopper-medium",
    "hopper-medium-replay-v2": "hopper-replay",
    "hopper-expert-v2": "hopper-expert",
    "halfcheetah-medium-v2": "hc-medium",
    "halfcheetah-medium-replay-v2": "hc-replay",
    "halfcheetah-expert-v2": "hc-expert",
    "walker2d-medium-v2": "w2d-medium",
    "walker2d-medium-replay-v2": "w2d-replay",
    "walker2d-expert-v2": "w2d-expert",
}
SEEDS = (2, 3)
ROOT = Path("/home/ext_csh/MPI_sweep")
SWEEPS = {
    "mpi1": (ROOT / "results" / "mpi1_s23", "mpi1", "d4rl_pi1"),
    "mpi2": (ROOT / "results" / "mpi2_s23", "mpi2", "d4rl_pi2"),
    "mpi3": (ROOT / "results" / "mpi3_s23", "mpi3", "d4rl_pi3"),
    "mpi8": (ROOT / "results" / "mpi8_s23", "mpi8", "d4rl_pi8"),
    "exp1": (ROOT / "results" / "exp1_s23", "exp1", "d4rl_pi1"),
    "exp2": (ROOT / "results" / "exp2_s23", "exp2", "d4rl_pi2"),
    "exp3": (ROOT / "results" / "exp3_s23", "exp3", "d4rl_pi3"),
}
CANVAS = Path(
    "/home/ext_csh/.cursor/projects/home-ext-csh/canvases/mpi-k123-s23.canvas.tsx"
)
KST = timezone(timedelta(hours=9))
BASE_TARGET = len(ENVS) * len(BASE_TAUS) * len(SEEDS)  # 252
HI_TARGET = len(ENVS) * len(HI_TAUS) * len(SEEDS)  # 72
TABLE_KEYS = ["mpi1", "mpi2", "mpi3", "exp2", "exp3"]
QUEUE_LOG = ROOT / "logs" / "exp_tau40_ext" / "queue_exp23_tau24_40.log"


def tau_key(t: float) -> str:
    return f"{t:g}"


def collect(root: Path, tag: str, score_key: str):
    cells: dict[tuple[str, str], dict[int, tuple[int, float]]] = defaultdict(dict)
    n_eval = 0
    n_1m = 0
    n_started = 0
    pat = re.compile(rf"(.+)_tau(.+)_{re.escape(tag)}_seed(\d+)$")
    if root.is_dir():
        n_started = sum(1 for _ in root.glob("*/config.json"))
        n_1m = sum(1 for _ in root.glob("*/params_1000000.pkl"))
    for path in root.glob("*/eval.csv") if root.is_dir() else []:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            continue
        last = rows[-1]
        match = pat.match(path.parent.name)
        if not match:
            continue
        raw = last.get(score_key) or last.get("d4rl_score")
        if not raw:
            continue
        env, tau, seed = match.group(1), match.group(2), int(match.group(3))
        if seed not in SEEDS:
            continue
        step = int(float(last["step"]))
        cells[(env, tau)][seed] = (step, float(raw))
        n_eval += 1
    matrix: dict[str, list[float | None]] = {}
    for env in ENVS:
        series: list[float | None] = []
        for t in TAUS:
            seeds = cells.get((env, tau_key(t)), {})
            if all(s in seeds and seeds[s][0] == 1_000_000 for s in SEEDS):
                vals = [seeds[s][1] for s in SEEDS]
                series.append(round(sum(vals) / len(vals), 2))
            else:
                series.append(None)
        matrix[env] = series
    return matrix, n_eval, n_1m, n_started


def hi_tau_1m_count(root: Path, tag: str) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for t in HI_TAUS:
        n += sum(
            1
            for _ in root.glob(f"*_tau{tau_key(t)}_{tag}_seed*/params_1000000.pkl")
        )
    return n


def hi_tau_by_env(root: Path, tag: str) -> list[int]:
    counts = []
    for env in ENVS:
        if not root.is_dir():
            counts.append(0)
            continue
        n = 0
        for t in HI_TAUS:
            n += sum(
                1
                for _ in root.glob(
                    f"{env}_tau{tau_key(t)}_{tag}_seed*/params_1000000.pkl"
                )
            )
        counts.append(n)
    return counts


def interpolate_only(vals: list):
    known = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(known) < 2:
        return list(vals)
    out: list[float | None] = [None] * len(vals)
    for (i, a), (j, b) in zip(known, known[1:]):
        out[i] = a
        out[j] = b
        span = j - i
        for k in range(i + 1, j):
            frac = (k - i) / span
            out[k] = round(a + (b - a) * frac, 2)
    return out


def js_series(vals):
    return "[" + ", ".join(str(0 if v is None else v) for v in vals) + "]"


def peak(vals):
    scored = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not scored:
        return None
    i, v = max(scored, key=lambda x: x[1])
    return tau_key(TAUS[i]), v


def chart_block(title: str, series_map: dict[str, list]):
    specs = [
        ("mpi1", "MPI π1", "info"),
        ("mpi2", "MPI π2", "success"),
        ("mpi3", "MPI π3", "warning"),
        ("exp2", "EXP π2", "neutral"),
        ("exp3", "EXP π3", "danger"),
    ]
    prepared = []
    bits = []
    for key, name, tone in specs:
        if key not in series_map:
            continue
        vals = series_map[key]
        if not any(v is not None for v in vals):
            continue
        filled = [0 if v is None else v for v in interpolate_only(vals)]
        prepared.append((name, filled, tone))
        pk = peak(vals)
        if pk:
            bits.append(f"{name} peak τ={pk[0]} ({pk[1]})")
    if not prepared:
        return None
    cats = ", ".join(f'"{tau_key(t)}"' for t in TAUS)
    series_js = []
    for name, filled, tone in prepared:
        series_js.append(
            f'{{ name: "{name}", data: {js_series(filled)}, tone: "{tone}" }}'
        )
    header = " · ".join(bits) if bits else title
    joined = ",\n              ".join(series_js)
    return f'''      <Card>
        <CardHeader trailing="{title}">{header}</CardHeader>
        <CardBody>
          <LineChart
            categories={{[{cats}]}}
            yMin={{0}}
            yMax={{110}}
            height={{240}}
            valueSuffix=" D4RL"
            series={{[
              {joined}
            ]}}
          />
        </CardBody>
      </Card>'''


def hi_progress_block(exp2_counts: list[int], exp3_counts: list[int]) -> str:
    cats = ", ".join(f'"{SHORT[e]}"' for e in ENVS)
    d2 = ", ".join(str(c) for c in exp2_counts)
    d3 = ", ".join(str(c) for c in exp3_counts)
    t2, t3 = sum(exp2_counts), sum(exp3_counts)
    return f'''      <Card>
        <CardHeader trailing="Exp τ∈{{24,28,34,40}}">1M cells · exp2 {t2}/{HI_TARGET} · exp3 {t3}/{HI_TARGET}</CardHeader>
        <CardBody>
          <BarChart
            categories={{[{cats}]}}
            yMin={{0}}
            yMax={{8}}
            height={{220}}
            valueSuffix=" cells"
            series={{[
              {{ name: "exp2", data: [{d2}], tone: "neutral" }},
              {{ name: "exp3", data: [{d3}], tone: "danger" }}
            ]}}
          />
        </CardBody>
      </Card>'''


def _cell(*vals):
    parts = ["0" if v is None else f"{v:.1f}" for v in vals]
    if all(p == "0" for p in parts):
        return '"0"'
    return f'"{" / ".join(parts)}"'


def live_counts() -> tuple[int, int, str, dict[str, int]]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,cmd"], text=True, errors="replace"
        )
    except Exception:
        return 0, 0, "unknown", {}
    trains = sum(1 for line in out.splitlines() if "train_td3bc.py" in line)
    sweeps = sum(1 for line in out.splitlines() if "launch_mpi_sweep.py" in line)
    phase = "idle"
    for line in out.splitlines():
        if "launch_mpi_sweep.py" not in line:
            continue
        hops = re.search(r"--hops\s+(\d+)", line)
        integ = re.search(r"--integrator\s+(\w+)", line)
        taus = re.search(r"--taus\s+(.+?)(?:\s+--|\s*$)", line)
        if hops and integ:
            phase = f"{integ.group(1)} K={hops.group(1)}"
            if taus:
                phase += f" τ={taus.group(1).strip()}"
            break
    qmeta: dict[str, int] = {"ok": 0, "fail": 0}
    if QUEUE_LOG.is_file():
        text = QUEUE_LOG.read_text(encoding="utf-8", errors="replace")
        qmeta["ok"] = len(re.findall(r"^\[ok\]", text, re.M))
        qmeta["fail"] = len(re.findall(r"^\[fail\]", text, re.M))
        if "[queue] phase=exp3" in text and "[queue] exp2 done" in text:
            if "launch_mpi_sweep.py" in out and "--hops 3" in out:
                phase = "explicit K=3 τ=24..40"
            elif trains == 0 and "[queue] exp3 done" in text:
                phase = "exp τ40 complete"
        elif "[queue] phase=exp2" in text:
            phase = phase if phase != "idle" else "explicit K=2 τ=24..40"
    return trains, sweeps, phase, qmeta


def tbl_rows(matrices: dict[str, dict[str, list]], keys: list[str], envs: list[str]):
    rows = []
    for env in envs:
        cells = [f'"{SHORT[env]}"']
        for i in range(len(TAUS)):
            cells.append(_cell(*(matrices[k][env][i] for k in keys)))
        rows.append("[" + ", ".join(cells) + "]")
    return ",\n          ".join(rows)


def render(
    matrices,
    stats,
    stamp: str,
    now: str,
    trains: int,
    sweeps: int,
    phase: str,
    qmeta: dict[str, int],
    exp2_hi: list[int],
    exp3_hi: list[int],
) -> str:
    charts = []
    for group, envs in (
        ("Hopper", ENVS[:3]),
        ("HalfCheetah", ENVS[3:6]),
        ("Walker2d", ENVS[6:]),
    ):
        blocks = []
        for env in envs:
            block = chart_block(
                SHORT[env],
                {k: matrices[k][env] for k in matrices},
            )
            if block:
                blocks.append(block)
        if blocks:
            charts.append(
                f"      <H2>{group}</H2>\n      <Grid columns={{1}} gap={{16}}>\n"
                + "\n".join(blocks)
                + "\n      </Grid>"
            )

    chart_section = "\n\n".join(charts) if charts else (
        '      <Callout tone="warning">\n'
        "        아직 seed 2·3 양쪽 1M이 끝난 τ 셀이 없어 곡선이 비어 있습니다.\n"
        "      </Callout>"
    )

    headers = ", ".join(f'"{tau_key(t)}"' for t in TAUS)
    n_pairs = {
        k: sum(1 for env in ENVS for v in matrices[k][env] if v is not None)
        for k in matrices
    }
    e2 = sum(exp2_hi)
    e3 = sum(exp3_hi)
    progress = hi_progress_block(exp2_hi, exp3_hi)
    total_hi = e2 + e3
    rem_hi = 2 * HI_TARGET - total_hi

    return f'''import {{
  Callout,
  Card,
  CardBody,
  CardHeader,
  BarChart,
  LineChart,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

/**
 * MPI π1–3 + EXP π2/π3 — seed-mean of {{2,3}} on ext_csh.
 * Live focus: Exp high-τ extension {{24,28,34,40}}.
 * Snapshot: {stamp}
 */

export default function MpiK123S23() {{
  return (
    <Stack gap={{24}}>
      <H1>MPI / EXP seed 2·3 — τ to 40</H1>
      <Text>
        ext_csh MPI_sweep · base τ≤20 plus high-τ {{24,28,34,40}} · phase: {phase}.
        Live queue: Exp K=2→K=3 high-τ (144 cells). Snapshot {now} KST.
      </Text>
      <Grid columns={{4}} gap={{16}}>
        <Stat value="{e2} / {HI_TARGET}" label="exp2 high-τ 1M" tone="neutral" />
        <Stat value="{e3} / {HI_TARGET}" label="exp3 high-τ 1M" tone="danger" />
        <Stat value="{trains} / {sweeps}" label="live trains / masters" />
        <Stat value="{qmeta.get('ok', 0)} ok / {qmeta.get('fail', 0)} fail" label="queue log" tone="info" />
      </Grid>
      <Grid columns={{3}} gap={{16}}>
        <Stat value="{stats['mpi1'][2]} / {BASE_TARGET}" label="mpi1 1M (base grid)" />
        <Stat value="{stats['mpi2'][2]} / {BASE_TARGET}" label="mpi2 1M" tone="success" />
        <Stat value="{stats['mpi3'][2]} / {BASE_TARGET}" label="mpi3 1M" tone="warning" />
      </Grid>
      <Callout tone="info">
        High-τ rem≈{rem_hi}/144. Curves: MPI π1–3 + EXP π2/π3. 표 칸: mpi1 / mpi2 / mpi3 / exp2 / exp3.
        seed-mean pairs: π1={n_pairs.get('mpi1', 0)} π2={n_pairs.get('mpi2', 0)} π3={n_pairs.get('mpi3', 0)}
        · exp2={n_pairs.get('exp2', 0)} exp3={n_pairs.get('exp3', 0)}.
      </Callout>

      <H2>Exp high-τ progress</H2>
{progress}

{chart_section}

      <H2>표 — mpi1 / mpi2 / mpi3 / exp2 / exp3</H2>
      <Text>칸 = seed-mean D4RL (없으면 0). Source: MPI_sweep/results/*_s23 · {stamp}</Text>
      <Table
        headers={{["env", {headers}]}}
        rows={{[
          {tbl_rows(matrices, TABLE_KEYS, ENVS)}
        ]}}
      />
    </Stack>
  );
}}
'''


def main() -> None:
    matrices = {}
    stats = {}
    for key, (root, tag, score_key) in SWEEPS.items():
        matrix, n_eval, n_1m, n_started = collect(root, tag, score_key)
        matrices[key] = matrix
        stats[key] = (n_eval, n_1m, n_1m, n_started)

    exp2_hi = hi_tau_by_env(*SWEEPS["exp2"][:2])
    exp3_hi = hi_tau_by_env(*SWEEPS["exp3"][:2])
    trains, sweeps, phase, qmeta = live_counts()
    now = datetime.now(KST)
    stamp = now.strftime("%Y-%m-%d %H:%M KST")
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(
        render(
            matrices,
            stats,
            stamp,
            now.strftime("%H:%M"),
            trains,
            sweeps,
            phase,
            qmeta,
            exp2_hi,
            exp3_hi,
        ),
        encoding="utf-8",
    )
    print(
        f"[canvas] {stamp} phase={phase} trains={trains} "
        f"exp2_hi={sum(exp2_hi)}/{HI_TARGET} exp3_hi={sum(exp3_hi)}/{HI_TARGET} "
        f"ok={qmeta.get('ok', 0)} fail={qmeta.get('fail', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
