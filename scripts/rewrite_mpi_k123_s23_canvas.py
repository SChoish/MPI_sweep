#!/usr/bin/env python3
"""Rewrite mpi-k123-s23.canvas.tsx from local MPI_sweep results (seeds 2,3)."""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

TAUS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0]
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
# Live mpi8 queue on ext_csh: halfcheetah + walker2d only.
MPI8_ENVS = ENVS[3:]
MPI8_TARGET = len(MPI8_ENVS) * len(TAUS) * len(SEEDS)  # 168
CANVAS = Path(
    "/home/ext_csh/.cursor/projects/home-ext-csh/canvases/mpi-k123-s23.canvas.tsx"
)
KST = timezone(timedelta(hours=9))
TARGET = len(ENVS) * len(TAUS) * len(SEEDS)  # 252
TABLE_KEYS = ["mpi1", "mpi2", "mpi3", "mpi8", "exp3"]


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


def env_1m_counts(root: Path, tag: str, envs: list[str]) -> list[int]:
    counts = []
    for env in envs:
        if not root.is_dir():
            counts.append(0)
            continue
        n = sum(
            1
            for _ in root.glob(f"{env}_tau*_{tag}_seed*/params_1000000.pkl")
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
        ("mpi8", "MPI π8", "neutral"),
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


def mpi8_progress_block(counts: list[int]) -> str:
    cats = ", ".join(f'"{SHORT[e]}"' for e in MPI8_ENVS)
    data = ", ".join(str(c) for c in counts)
    total = sum(counts)
    return f'''      <Card>
        <CardHeader trailing="mpi8 queue">1M cells by env · {total} / {MPI8_TARGET} (hc+w × 14τ × seeds 2·3)</CardHeader>
        <CardBody>
          <BarChart
            categories={{[{cats}]}}
            yMin={{0}}
            yMax={{28}}
            height={{220}}
            valueSuffix=" cells"
            series={{[
              {{ name: "1M complete", data: [{data}], tone: "info" }}
            ]}}
          />
        </CardBody>
      </Card>'''


def _cell(*vals):
    parts = ["0" if v is None else f"{v:.1f}" for v in vals]
    if all(p == "0" for p in parts):
        return '"0"'
    return f'"{" / ".join(parts)}"'


def live_counts() -> tuple[int, int, str]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,cmd"], text=True, errors="replace"
        )
    except Exception:
        return 0, 0, "unknown"
    trains = sum(1 for line in out.splitlines() if "train_td3bc.py" in line)
    sweeps = sum(1 for line in out.splitlines() if "launch_mpi_sweep.py" in line)
    phase = "idle"
    for line in out.splitlines():
        if "launch_mpi_sweep.py" not in line:
            continue
        hops = re.search(r"--hops\s+(\d+)", line)
        integ = re.search(r"--integrator\s+(\w+)", line)
        if hops and integ:
            phase = f"{integ.group(1)} K={hops.group(1)}"
            break
    chain_dir = ROOT / "logs" / "_chain"
    logs = sorted(
        chain_dir.glob("chain_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if phase == "idle" and logs:
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
        starts = re.findall(r"START integrator=(\w+) hops=(\d+)", text)
        if starts:
            integ, hops = starts[-1]
            phase = f"{integ} K={hops}"
        if trains == 0 and "CHAIN_COMPLETE" in text:
            phase = "complete"
    # Prefer live mpi8 chain log if present.
    mpi8_logs = sorted(
        chain_dir.glob("launch_mpi8*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if trains == 0 and mpi8_logs:
        text = mpi8_logs[-1].read_text(encoding="utf-8", errors="replace")
        if "CHAIN_COMPLETE" in text:
            phase = "complete"
    return trains, sweeps, phase


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
    mpi8_counts: list[int],
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
    mpi8_done = stats["mpi8"][2]
    progress = mpi8_progress_block(mpi8_counts)

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
 * MPI π1–3 / π8 / EXP π3 — seed-mean of {{2,3}} on ext_csh.
 * Snapshot: {stamp}
 */

export default function MpiK123S23() {{
  return (
    <Stack gap={{24}}>
      <H1>MPI π1–3 / π8 + EXP π3 — seed 2·3 mean D4RL</H1>
      <Text>
        ext_csh MPI_sweep · τ≤20 (14) · concurrency 8 · phase: {phase}.
        Live queue: mpi8 halfcheetah+walker2d only ({MPI8_TARGET} cells).
        미완료 칸은 0. 알려진 1M 점 사이만 보간. 스냅샷 {now} KST.
      </Text>
      <Grid columns={{4}} gap={{16}}>
        <Stat value="{mpi8_done} / {MPI8_TARGET}" label="mpi8 1M (hc+w)" tone="info" />
        <Stat value="{trains} / {sweeps}" label="live trains / sweep masters" />
        <Stat value="{stats['mpi3'][2]} / {TARGET}" label="mpi3 1M cells" tone="warning" />
        <Stat value="{stats['exp3'][2]} / {TARGET}" label="exp3 1M cells" tone="danger" />
      </Grid>
      <Grid columns={{3}} gap={{16}}>
        <Stat value="{stats['mpi1'][2]} / {TARGET}" label="mpi1 1M cells" />
        <Stat value="{stats['mpi2'][2]} / {TARGET}" label="mpi2 1M cells" tone="success" />
        <Stat value="{n_pairs['mpi8']}" label="mpi8 seed-mean τ pairs" />
      </Grid>
      <Callout tone="info">
        곡선: MPI π1/π2/π3/π8 + EXP π3. 표 칸: mpi1 / mpi2 / mpi3 / mpi8 / exp3.
        seed-mean 쌍: π1={n_pairs['mpi1']} π2={n_pairs['mpi2']} π3={n_pairs['mpi3']}
        π8={n_pairs['mpi8']} · exp3={n_pairs['exp3']}.
      </Callout>

      <H2>MPI π8 progress</H2>
{progress}

{chart_section}

      <H2>표 — mpi1 / mpi2 / mpi3 / mpi8 / exp3</H2>
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

    mpi8_root, mpi8_tag, _ = SWEEPS["mpi8"]
    mpi8_counts = env_1m_counts(mpi8_root, mpi8_tag, MPI8_ENVS)

    trains, sweeps, phase = live_counts()
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
            mpi8_counts,
        ),
        encoding="utf-8",
    )
    keys = ("mpi1", "mpi2", "mpi3", "mpi8", "exp1", "exp2", "exp3")
    print(
        f"[canvas] {stamp} phase={phase} trains={trains} "
        + " ".join(f"{k}_1m={stats[k][2]}" for k in keys)
        + f" mpi8_scope={stats['mpi8'][2]}/{MPI8_TARGET}",
        flush=True,
    )


if __name__ == "__main__":
    main()
