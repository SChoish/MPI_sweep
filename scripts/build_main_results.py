#!/usr/bin/env python3
"""Build MAIN_RESULTS.md from published final scores; Python standard library only."""
import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"awr_gaussian_fr": "AWR–FR", "qbc_gaussian_w2": "Gaussian Q+BC–W2"}
ENVS = [f"{d}-{s}-v2" for d in ("hopper", "halfcheetah", "walker2d")
        for s in ("medium", "medium-replay", "expert")]
GRID = [.05, .1, .2, .4, .7, 1.25, 1.5, 2.5, 4, 7, 10]
# These two legacy publishers omit step/completion in their final-score exports.
LEGACY_FINAL = {"iql_gauss_v5", "iql_k4_t6"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("non-finite value")
    return x


def machine(row, status):
    for obj in (row, status):
        for key in ("machine", "hostname", "host"):
            if obj.get(key):
                return str(obj[key])
    # Account provenance is weaker than a measured hostname: retain that distinction.
    m = re.search(r"/home/([^/]+)/", str(row.get("source", "") or status.get("source", "")))
    return m.group(1) + " (계정)" if m else "미확인"


def collect(root):
    candidates = defaultdict(list)
    warnings = set()
    sources = set()
    planned = set()

    def add(section, env, t, k, seed, score, host, path, line, priority=0):
        if k not in (1, 2, 3, 4) or seed not in (0, 1, 2, 3):
            return
        if t <= 0:
            raise ValueError("T must be positive")
        key = (section, env, t, k, seed)
        candidates[key].append(dict(score=score, machine=host, source=str(path.relative_to(root)),
                                    line=line, priority=priority))
        sources.add(str(path.relative_to(root)))

    for folder in sorted((root / "sweep_results").glob("iql*")):
        status = read_json(folder / "STATUS.json")
        for path in sorted(folder.glob("scores*.csv")):
            with path.open(newline="") as f:
                for line, row in enumerate(csv.DictReader(f), 2):
                    variant = row.get("variant") or status.get("variant")
                    if variant not in VARIANTS:
                        continue
                    expected = "fr" if variant == "awr_gaussian_fr" else "w2"
                    geometry = row.get("geometry") or status.get("geometry") or expected
                    if geometry != expected:
                        continue
                    mode = row.get("gaussian_qbc_mode") or status.get("gaussian_qbc_mode", "stochastic")
                    if variant == "qbc_gaussian_w2" and mode != "stochastic":
                        continue
                    if row.get("eval_mode") != "mean":
                        continue
                    section = VARIANTS[variant]
                    env = row.get("env", "")
                    if not env:
                        warnings.add(f"{path.relative_to(root)}: env 누락")
                        continue
                    try:
                        t = number(row.get("tau") or row.get("T"))
                        k = int(row["K"])
                        hop = int(row.get("hop", k))
                        seed = int(row["seed"])
                        policy = row.get("policy", "")
                        priority = 0
                        if policy == "baseline" and k > 1:
                            # v6 baseline is at h, not T. Never relabel that as K=1 at T.
                            if folder.name != "iql_k4_t6" or status.get("schema") != "iql_actor_geometry_v5_consistent_gaussian":
                                continue
                            k, priority = 1, -1
                        elif k == 1 and policy in ("baseline", "mpi") and hop == 1:
                            priority = 2
                        elif policy != "mpi" or hop != k:
                            continue
                        elif folder.name == "iql_k4_t6":
                            priority = 1  # canonical shared-bank K4 over legacy pilot duplicates
                        planned.add((section, env, t))
                        state = row.get("status", "").lower()
                        if state and state not in ("completed", "complete", "done", "available"):
                            continue
                        step = row.get("step") or row.get("timesteps")
                        if step and number(step) != 1_000_000:
                            continue
                        if not step and folder.name not in LEGACY_FINAL:
                            warnings.add(f"{path.relative_to(root)}: step 없는 신규 소스는 집계 보류")
                            continue
                        score = number(row["d4rl_score"])
                        add(section, env, t, k, seed, score, machine(row, status), path, line, priority)
                    except (ValueError, TypeError, KeyError):
                        warnings.add(f"{path.relative_to(root)}:{line}: 필수 필드/수치 확인 필요")

    # Read canonical matrices directly, not potentially stale scores_long.csv.
    for path in sorted((root / "sweep_results").glob("K=*/*/seed*.csv")):
        m = re.fullmatch(r"K=([1-4])/(Imp|Exp)/seed([0-3])\.csv", str(path.relative_to(root / "sweep_results")))
        if not m:
            continue
        k, integ, seed = int(m[1]), m[2], int(m[3])
        section = "TD3+BC MPI · " + ("Implicit" if integ == "Imp" else "Explicit")
        provenance = read_json(path.with_suffix(".provenance.json"))
        with path.open(newline="") as f:
            for line, row in enumerate(csv.DictReader(f), 2):
                t = number(row["tau"])
                for env in ENVS:
                    planned.add((section, env, t))
                    raw = row.get(env, "").strip()
                    if not raw:
                        continue
                    try:
                        host = provenance.get("cells", {}).get(f"{t:g}/{env}", {}).get("machine") or "미확인"
                        add(section, env, t, k, seed, number(raw), host, path, line)
                    except ValueError:
                        warnings.add(f"{path.relative_to(root)}:{line}: {env} 비유한 수치")

    selected = {}
    audit = []
    for key, rows in sorted(candidates.items()):
        # Never pick the highest score. Conflicts with equal priority are withheld.
        rank = max(r["priority"] for r in rows)
        top = [r for r in rows if r["priority"] == rank]
        conflict = any(abs(r["score"] - top[0]["score"]) > 1e-8 for r in top)
        chosen = None if conflict else min(top, key=lambda r: (r["source"], r["line"]))
        if conflict:
            warnings.add(f"중복 충돌 보류: {key}")
        else:
            selected[key] = chosen
        for r in rows:
            audit.append(dict(zip(("method", "env", "T", "K", "seed"), key), **r,
                              selected=r is chosen, conflict=conflict))
    return selected, planned, sorted(warnings), sorted(sources), audit


def cell(rows):
    if not rows:
        return "—"
    vals = [r["score"] for _, r in rows]
    score = f"{statistics.mean(vals):.2f}"
    if len(vals) > 1:
        score += f" ± {statistics.stdev(vals):.2f}"
    detail = []
    for seed, r in rows:
        label = html.escape(f"s{seed}: {r['score']:.2f} · {r['machine']}").replace("|", "&#124;")
        detail.append(f"[{label}]({quote(r['source'], safe='/')}#L{r['line']})")
    return f"**{score}** ({len(vals)}/4)" + "<br>" + "<br>".join(detail)


def visible_results(selected):
    """TD3 rows require all four K values within the same environment/integrator."""
    return {key: value for key, value in selected.items()
            if not key[0].startswith("TD3+BC MPI") or all(
                any((*key[:3], k, seed) in selected for seed in range(4))
                for k in range(1, 5))}


def build(root):
    selected, planned, warnings, sources, audit = collect(root)
    selected = visible_results(selected)
    sections = [*VARIANTS.values(), "TD3+BC MPI · Implicit", "TD3+BC MPI · Explicit"]
    out = ["# Main experiment results", "",
           "자동 생성: `python3 scripts/build_main_results.py`. 방법 → 환경 → T 순서이며 K=1~4를 같은 표에서 비교합니다.", "",
           "- 각 칸: **정규화 점수 평균 ± 표본 표준편차(ddof=1)**, 확보한 시드 수, 시드별 점수·실행 머신·원본 링크. 시드는 0~3입니다.",
           "- 4시드 미만의 평균은 진행 중 참고값입니다. n=1은 표준편차를 표시하지 않습니다. `—`는 채택 가능한 게시 점수가 없다는 뜻이며 실행 중 여부를 뜻하지 않습니다.",
           "- IQL은 mean-action 평가의 최종 actor만 사용합니다. 기존 두 publisher의 step 없는 CSV는 최종 점수 export로 취급하며, 신규 소스는 step=1000000이 필요합니다. 원격 체크포인트 존재까지 검증하는 표는 아닙니다.",
           "- K=1은 standalone 결과 우선. shchoi v5의 full-T baseline은 보완에만 사용합니다. v6의 h-baseline과 중간 actor는 제외합니다. K=4도 머신과 무관하게 같은 표에 합칩니다.",
           "- 중복은 점수 최대값으로 고르지 않습니다. standalone K1 / shared-bank K4 우선순위 적용 후 동순위 점수 충돌은 보류합니다. 전체 후보는 [집계 감사 JSON](reports/main_results_audit.json)에 기록합니다.",
           "- 머신은 명시된 hostname/machine 우선입니다. STATUS의 `/home/<account>`만 있으면 `(계정)`으로 표시합니다. 과거 TD3+BC 행렬에는 머신 정보가 없어 `미확인`으로 남깁니다.",
           "- TD3+BC MPI는 같은 integrator·환경·T에서 K=1~4가 모두 있는 행만 표시합니다. 현재 Explicit는 이 조건을 충족하지 않아 생략하며, 원본과 감사 JSON은 유지합니다.",
           "- T는 총 horizon, h=T/K. K=1 기준 AWR 역온도=T, Gaussian BC 계수=1/T. TD3+BC implicit actor 계수 α=2T(Q 정규화 적용).", "",
           "| 방법 | 게시된 시드 점수 | 4시드 확보 셀 |", "|---|---:|---:|"]
    for section in sections:
        keys = [k for k in selected if k[0] == section]
        if section.startswith("TD3+BC MPI") and not keys:
            continue
        groups = defaultdict(int)
        for key in keys:
            groups[key[:-1]] += 1
        out.append(f"| {section} | {len(keys)} | {sum(n == 4 for n in groups.values())} |")
    for section in sections:
        if section.startswith("TD3+BC MPI") and not any(k[0] == section for k in selected):
            continue
        out += ["", f"## {section}", ""]
        if section.startswith("TD3+BC MPI"):
            horizons = sorted({k[2] for k in selected if k[0] == section})
            out += ["K=1~4 모두 게시된 환경·T만 표시합니다. T: " + ", ".join(f"{t:g}" for t in horizons) + ".", ""]
        envs = list(ENVS) + sorted({e for s, e, _ in planned if s == section} - set(ENVS))
        for env in envs:
            ts = {t for s, e, t in planned if s == section and e == env}
            if section in VARIANTS.values():
                ts |= set(GRID)
            if section.startswith("TD3+BC MPI"):
                ts = {t for t in ts if any(k[:3] == (section, env, t) for k in selected)}
            if not ts:
                continue
            out += ["<details open>" if any(k[0] == section and k[1] == env for k in selected) else "<details>",
                    f"<summary>{env}</summary>", "", "| T | K=1 | K=2 | K=3 | K=4 |", "|---:|---|---|---|---|"]
            for t in sorted(ts):
                cells = [cell([(s, selected[(section, env, t, k, s)]) for s in range(4)
                               if (section, env, t, k, s) in selected]) for k in range(1, 5)]
                out.append(f"| {t:g} | " + " | ".join(cells) + " |")
            out += ["", "</details>", ""]
    out += ["## 소스·집계 점검", "", "새 IQL 소스는 `sweep_results/iql*/scores*.csv`에서 자동 발견합니다. variant, geometry, eval_mode, policy, hop, K, tau/T, seed, step, d4rl_score를 명시하세요. STATUS.json의 variant·geometry·source·machine도 읽습니다.", "",
            "TD3+BC는 `sweep_results/K=<K>/<Imp|Exp>/seed<seed>.csv`를 직접 읽습니다. 셀별 머신은 같은 이름의 `.provenance.json`에 `cells: {\"0.05/hopper-medium-v2\": {\"machine\": \"host\"}}`로 추가할 수 있습니다.", "",
            "GitHub Actions는 main의 결과 push에 반응합니다. 다른 Actions가 GITHUB_TOKEN으로 push한 경우 push 이벤트가 재발화되지 않을 수 있어 시간별 재집계도 수행합니다.", ""]
    out += [f"- [{s}]({quote(s, safe='/')})" for s in sources]
    out += ["", "### 확인 필요", ""] + (["- " + x for x in warnings] or ["현재 파싱 경고 없음."])
    (root / "MAIN_RESULTS.md").write_text("\n".join(out) + "\n")
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/main_results_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"MAIN_RESULTS.md: {len(selected)} seed scores, {len(warnings)} warnings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    build(parser.parse_args().root)
