#!/usr/bin/env python3
"""Rewrite mpi-k123-s23.canvas.tsx from local results ∪ git CSV (any-seed mean)."""

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
SEEDS_PLOT = (0, 1, 2, 3)  # average whatever seeds exist per cell
SEEDS_QUEUE = (2, 3)  # live Imp high-τ queue progress only
ROOT = Path("/home/ext_csh/MPI_sweep")
SWEEPS = {
    "mpi1": (ROOT / "results" / "mpi1_s23", "mpi1", "d4rl_pi1"),
    "mpi2": (ROOT / "results" / "mpi2_s23", "mpi2", "d4rl_pi2"),
    "mpi3": (ROOT / "results" / "mpi3_s23", "mpi3", "d4rl_pi3"),
    "mpi4": (ROOT / "results" / "mpi4_s23", "mpi4", "d4rl_pi4"),
    "mpi8": (ROOT / "results" / "mpi8_s23", "mpi8", "d4rl_pi8"),
    "exp1": (ROOT / "results" / "exp1_s23", "exp1", "d4rl_pi1"),
    "exp2": (ROOT / "results" / "exp2_s23", "exp2", "d4rl_pi2"),
    "exp3": (ROOT / "results" / "exp3_s23", "exp3", "d4rl_pi3"),
    "exp4": (ROOT / "results" / "exp4_s23", "exp4", "d4rl_pi4"),
}
CANVAS = Path(
    "/home/ext_csh/.cursor/projects/home-ext-csh/canvases/mpi-k123-s23.canvas.tsx"
)
KST = timezone(timedelta(hours=9))
BASE_TARGET = len(ENVS) * len(BASE_TAUS) * len(SEEDS_QUEUE)  # 252
HI_TARGET = len(ENVS) * len(HI_TAUS) * len(SEEDS_QUEUE)  # 72
TABLE_KEYS = ["mpi1", "mpi2", "mpi3", "mpi4", "exp2", "exp3"]
QUEUE_LOG_DIR = ROOT / "logs" / "mpi_tau40_ext"
CSV_ROOT = ROOT / "sweep_results"
# canvas key -> (K, Imp|Exp) under sweep_results/ (git-tracked scores)
CSV_KEYS = {
    "mpi1": (1, "Imp"),
    "mpi2": (2, "Imp"),
    "mpi3": (3, "Imp"),
    "mpi4": (4, "Imp"),
    "mpi8": (8, "Imp"),
    "exp1": (1, "Exp"),
    "exp2": (2, "Exp"),
    "exp3": (3, "Exp"),
    "exp4": (4, "Exp"),
}


def latest_queue_log() -> Path:
    cands = sorted(QUEUE_LOG_DIR.glob("queue_*.log"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else QUEUE_LOG_DIR / "queue_mpi234_tau24_40.log"


def tau_key(t: float) -> str:
    return f"{t:g}"


def mean_or_none(vals: list[float]) -> float | None:
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def cells_to_matrix(
    cells: dict[tuple[str, str], dict[int, float]],
) -> dict[str, list[float | None]]:
    matrix: dict[str, list[float | None]] = {}
    for env in ENVS:
        series: list[float | None] = []
        for t in TAUS:
            series.append(mean_or_none(list(cells.get((env, tau_key(t)), {}).values())))
        matrix[env] = series
    return matrix


def collect_local_cells(
    root: Path, tag: str, score_key: str
) -> tuple[dict[tuple[str, str], dict[int, float]], int, int, int]:
    """(env, tau) -> {seed: score} from local 1M evals; any seed."""
    cells: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
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
        step = int(float(last["step"]))
        if step != 1_000_000:
            continue
        if not (path.parent / "params_1000000.pkl").is_file():
            continue
        cells[(env, tau)][seed] = float(raw)
        n_eval += 1
    return cells, n_eval, n_1m, n_started


def collect_csv_cells(k: int, integ: str) -> dict[tuple[str, str], dict[int, float]]:
    """(env, tau) -> {seed: score} from git CSV seeds 0–3; skip empty cells."""
    cells: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for seed in SEEDS_PLOT:
        path = CSV_ROOT / f"K={k}" / integ / f"seed{seed}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw_tau = (row.get("tau") or "").strip()
                if not raw_tau:
                    continue
                tk = tau_key(float(raw_tau))
                for env in ENVS:
                    raw = (row.get(env) or "").strip()
                    if raw:
                        cells[(env, tk)][seed] = float(raw)
    return cells


def merge_cells(
    local: dict[tuple[str, str], dict[int, float]],
    csv_cells: dict[tuple[str, str], dict[int, float]],
) -> dict[tuple[str, str], dict[int, float]]:
    """Union seeds; local overrides the same seed id."""
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    keys = set(local) | set(csv_cells)
    for key in keys:
        merged = dict(csv_cells.get(key, {}))
        merged.update(local.get(key, {}))
        out[key] = merged
    return out


def count_filled(matrix: dict[str, list[float | None]]) -> int:
    return sum(1 for env in ENVS for v in matrix[env] if v is not None)


def seed_count_summary(
    cells: dict[tuple[str, str], dict[int, float]],
) -> str:
    """e.g. n1=12 n2=108 n3=0 n4=36 for how many cells used 1/2/3/4 seeds."""
    hist = defaultdict(int)
    for env in ENVS:
        for t in TAUS:
            n = len(cells.get((env, tau_key(t)), {}))
            if n:
                hist[n] += 1
    if not hist:
        return "none"
    return " ".join(f"n{k}={hist[k]}" for k in sorted(hist))


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
        ("mpi4", "MPI π4", "info"),
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


def hi_progress_block(
    mpi2_counts: list[int], mpi3_counts: list[int], mpi4_counts: list[int]
) -> str:
    cats = ", ".join(f'"{SHORT[e]}"' for e in ENVS)
    d2 = ", ".join(str(c) for c in mpi2_counts)
    d3 = ", ".join(str(c) for c in mpi3_counts)
    d4 = ", ".join(str(c) for c in mpi4_counts)
    t2, t3, t4 = sum(mpi2_counts), sum(mpi3_counts), sum(mpi4_counts)
    return f'''      <Card>
        <CardHeader trailing="Imp τ∈{{24,28,34,40}}">1M · mpi2 {t2}/{HI_TARGET} · mpi3 {t3}/{HI_TARGET} · mpi4 {t4}/{HI_TARGET}</CardHeader>
        <CardBody>
          <BarChart
            categories={{[{cats}]}}
            yMin={{0}}
            yMax={{8}}
            height={{220}}
            valueSuffix=" cells"
            series={{[
              {{ name: "mpi2", data: [{d2}], tone: "success" }},
              {{ name: "mpi3", data: [{d3}], tone: "warning" }},
              {{ name: "mpi4", data: [{d4}], tone: "info" }}
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
    queue_log = latest_queue_log()
    if queue_log.is_file():
        text = queue_log.read_text(encoding="utf-8", errors="replace")
        qmeta["ok"] = len(re.findall(r"^\[ok\]", text, re.M))
        qmeta["fail"] = len(re.findall(r"^\[fail\]", text, re.M))
        if "[queue] phase=mpi4" in text:
            phase = "implicit K=4 τ=24..40" if sweeps else (
                "mpi4 τ40 complete" if "[queue] mpi4 done" in text else phase
            )
        elif "[queue] phase=mpi3" in text:
            phase = "implicit K=3 τ=24..40" if sweeps else (
                "mpi3 τ40 complete" if "[queue] mpi3 done" in text else phase
            )
        elif "[queue] phase=mpi2" in text:
            phase = "implicit K=2 τ=24..40" if sweeps or trains else phase
        if "[queue] all done" in text and trains == 0:
            phase = "mpi2–4 τ40 complete"
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
    mpi2_hi: list[int],
    mpi3_hi: list[int],
    mpi4_hi: list[int],
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
        "        아직 어떤 seed에서도 1M 점수가 없어 곡선이 비어 있습니다.\n"
        "      </Callout>"
    )

    headers = ", ".join(f'"{tau_key(t)}"' for t in TAUS)
    n_pairs = {
        k: sum(1 for env in ENVS for v in matrices[k][env] if v is not None)
        for k in matrices
    }
    m2 = sum(mpi2_hi)
    m3 = sum(mpi3_hi)
    m4 = sum(mpi4_hi)
    progress = hi_progress_block(mpi2_hi, mpi3_hi, mpi4_hi)
    rem_hi = 3 * HI_TARGET - (m2 + m3 + m4)

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
 * MPI π1–4 + EXP π2–3 — available-seed mean (any of 0–3) on ext_csh.
 * Live focus: Imp (mpi2/3/4) high-τ extension {{24,28,34,40}}.
 * Snapshot: {stamp}
 */

export default function MpiK123S23() {{
  return (
    <Stack gap={{24}}>
      <H1>MPI / EXP — available-seed mean · τ to 40</H1>
      <Text>
        ext_csh MPI_sweep · 셀마다 있는 seed만 평균 (0–3, 개수 무관) · phase: {phase}.
        Live queue: mpi2→mpi3→mpi4 seed{{2,3}} high-τ. Snapshot {now} KST.
      </Text>
      <Grid columns={{4}} gap={{16}}>
        <Stat value="{m2} / {HI_TARGET}" label="mpi2 high-τ 1M" tone="success" />
        <Stat value="{m3} / {HI_TARGET}" label="mpi3 high-τ 1M" tone="warning" />
        <Stat value="{m4} / {HI_TARGET}" label="mpi4 high-τ 1M" tone="info" />
        <Stat value="{trains} / {sweeps}" label="live trains / masters" />
      </Grid>
      <Grid columns={{3}} gap={{16}}>
        <Stat value="{stats['mpi2'][2]} / {BASE_TARGET}" label="mpi2 1M (incl base)" tone="success" />
        <Stat value="{stats['mpi3'][2]} / {BASE_TARGET}" label="mpi3 1M (incl base)" tone="warning" />
        <Stat value="{stats.get('mpi4', (0,0,0,0))[2]} / {BASE_TARGET}" label="mpi4 1M (incl base)" tone="info" />
      </Grid>
      <Callout tone="info">
        Imp high-τ rem≈{rem_hi}/216 · queue {qmeta.get('ok', 0)} ok / {qmeta.get('fail', 0)} fail.
        표 칸: mpi1 / mpi2 / mpi3 / mpi4 / exp2 / exp3.
        filled cells: mpi1={n_pairs.get('mpi1', 0)} mpi2={n_pairs.get('mpi2', 0)} mpi3={n_pairs.get('mpi3', 0)} mpi4={n_pairs.get('mpi4', 0)} exp2={n_pairs.get('exp2', 0)} exp3={n_pairs.get('exp3', 0)}.
      </Callout>

      <H2>Imp high-τ progress</H2>
{progress}

{chart_section}

      <H2>표 — mpi1 / mpi2 / mpi3 / mpi4 / exp2 / exp3</H2>
      <Text>칸 = available-seed mean D4RL (없으면 0). Source: results ∪ git sweep_results seed0–3 · {stamp}</Text>
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
    seed_hist = {}
    for key, (root, tag, score_key) in SWEEPS.items():
        local_cells, n_eval, n_1m, n_started = collect_local_cells(root, tag, score_key)
        if key in CSV_KEYS:
            k, integ = CSV_KEYS[key]
            csv_cells = collect_csv_cells(k, integ)
            cells = merge_cells(local_cells, csv_cells)
        else:
            cells = local_cells
        matrices[key] = cells_to_matrix(cells)
        stats[key] = (n_eval, n_1m, n_1m, n_started)
        seed_hist[key] = seed_count_summary(cells)

    mpi2_hi = hi_tau_by_env(*SWEEPS["mpi2"][:2])
    mpi3_hi = hi_tau_by_env(*SWEEPS["mpi3"][:2])
    mpi4_hi = hi_tau_by_env(*SWEEPS["mpi4"][:2])
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
            mpi2_hi,
            mpi3_hi,
            mpi4_hi,
        ),
        encoding="utf-8",
    )
    merged_n = {k: count_filled(matrices[k]) for k in TABLE_KEYS}
    print(
        f"[canvas] {stamp} phase={phase} trains={trains} "
        f"mpi2_hi={sum(mpi2_hi)}/{HI_TARGET} mpi3_hi={sum(mpi3_hi)}/{HI_TARGET} "
        f"mpi4_hi={sum(mpi4_hi)}/{HI_TARGET} "
        f"ok={qmeta.get('ok', 0)} fail={qmeta.get('fail', 0)} "
        f"merged={merged_n} seeds={ {k: seed_hist.get(k, '') for k in TABLE_KEYS} }",
        flush=True,
    )


if __name__ == "__main__":
    main()