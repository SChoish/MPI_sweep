#!/usr/bin/env python3
"""Inventory published IQL logs using the main-results score selection."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from build_main_results import ROOT, VARIANTS, collect, machine, read_json

LABELS = {**VARIANTS, "qbc_deterministic_w2": "Deterministic Q+BC–W2 (본문 제외)"}
OTHER = "Gaussian Q+BC–FR (본문 제외)"


def score_files(folder):
    return sorted(folder.glob("scores*.csv"))


def methods(folder, status):
    variants = set(status.get("variants") or [])
    if status.get("variant"):
        variants.add(status["variant"])
    for path in score_files(folder):
        with path.open(newline="") as stream:
            variants.update(row["variant"] for row in csv.DictReader(stream) if row.get("variant"))
    if folder.name == "iql_gauss_fr_loco":
        return [OTHER]
    if status.get("geometry") == "fr" and "qbc_gaussian_w2" in variants:
        variants.remove("qbc_gaussian_w2")
        return sorted([OTHER] + [LABELS.get(v, v) for v in variants])
    return sorted(LABELS.get(v, v) for v in variants) or ["미표기"]


def csv_info(folder):
    files, hosts, accounts = [], set(), set()
    for path in score_files(folder):
        with path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            fields, count = rows.fieldnames or [], 0
            for row in rows:
                count += 1
                if row.get("hostname"):
                    hosts.add(row["hostname"])
                if row.get("machine"):
                    accounts.add(row["machine"])
            files.append(dict(path=path.name, rows=count, columns=fields))
    names = sorted(accounts | hosts)
    return files, names


def inventory(root):
    selected, _, warnings, _, audit = collect(root)
    chosen, candidates = defaultdict(list), Counter()
    for row in audit:
        parts = row["source"].split("/")
        if len(parts) < 3 or parts[0] != "sweep_results" or not parts[1].startswith("iql"):
            continue
        source = parts[1]
        candidates[source] += 1
        if row["selected"]:
            chosen[source].append(row)
    entries = []
    for folder in sorted((root / "sweep_results").glob("iql*")):
        if not folder.is_dir():
            continue
        status = read_json(folder / "STATUS.json")
        files, hosts = csv_info(folder)
        complete = status.get("done")
        if complete is None:
            complete = status.get("counts", {}).get("complete")
        status_file = folder / "STATUS.json"
        export = folder / "EXPORT.json"
        location = status.get("source") or status.get("save_dir") or ""
        account = machine({}, status)
        if account == "미확인":
            match = re.search(r"/(?:home|raid)/([^/]+)/", location)
            if match:
                account = match[1] + " (계정)"
        entries.append(dict(
            source=str(folder.relative_to(root)), methods=methods(folder, status),
            machine=", ".join(hosts) if hosts else account,
            updated_at=status.get("updated_at") or status.get("stamp_kst"),
            complete_reported=complete,
            planned_reported=status.get("total", status.get("planned")),
            csv_files=files, csv_rows=sum(f["rows"] for f in files),
            main_candidates=candidates[folder.name], main_selected=len(chosen[folder.name]),
            selected_by_method=dict(sorted(Counter(r["method"] for r in chosen[folder.name]).items())),
            export_manifest=str(export.relative_to(root)) if export.exists() else None,
            status_file=str(status_file.relative_to(root)) if status_file.exists() else None))
    by_method = Counter(k[0] for k in selected if k[0] in VARIANTS.values())
    return dict(schema="iql_log_index_v1", sources=entries,
                main_selected_by_method=dict(sorted(by_method.items())),
                main_parser_warnings=warnings)


def render(data):
    def link(path):
        relative = path.removeprefix("sweep_results/")
        return f"[{Path(path).name}]({relative})"

    out = [
        "# IQL 게시 로그 안내", "",
        "자동 생성: python3 scripts/build_log_index.py. 원본 CSV와 상태 파일의 위치는 유지합니다.",
        "상태 완료 수는 게시자가 보고한 진행 상황입니다. 본문 채택 수는 build_main_results.py의 최종 actor·mean 평가·중복 검사 결과입니다.",
        "",
        "| 게시 폴더 | 실험 | 머신·출처 | 상태 완료/계획 | CSV 원시 행 | 본문 채택 | 상태 갱신 |",
        "|---|---|---|---:|---:|---:|---|"]
    for item in data["sources"]:
        source, files = item["source"], item["csv_files"]
        path = item["status_file"] or (source + "/" + files[0]["path"] if files else None)
        label = link(path) if path else source
        done, planned = item["complete_reported"], item["planned_reported"]
        progress = f"{done if done is not None else '—'}/{planned if planned is not None else '—'}"
        out.append(f"| {label} | {', '.join(item['methods'])} | {item['machine']} | "
                   f"{progress} | {item['csv_rows']} | {item['main_selected']} | "
                   f"{item['updated_at'] or '—'} |")
    out += ["", "## 파일 역할", "",
            "- STATUS.json: 머신이 보고한 계획·진행 상태. 완료 수만으로 점수를 만들지 않습니다.",
            "- scores_verified.csv + EXPORT.json: 게시자의 검증된 최종점수 export와 근거.",
            "- scores.csv / scores_long.csv: 원래 게시 형식. 중간 actor, sample 평가, 다른 변형이 섞일 수 있습니다.",
            "- queue.log: 큐의 실행 기록. 점수나 학습 완료의 근거로 사용하지 않습니다.",
            "- [MAIN_RESULTS.md](../MAIN_RESULTS.md): 채택 점수. [main_results_audit.json](../reports/main_results_audit.json): 후보별 선택·충돌·출처.",
            "- 상태 완료는 run 수이고 본문 채택은 방법·환경·T·K·시드별 점수 수입니다. 공유 K=4 run에 여러 방법의 점수와 구형 full-T K=1 baseline이 있으면 본문 채택 수가 완료 run 수보다 클 수 있습니다.",
            "", "## 파일별 원시 행과 채택 점수", "",
            "| 게시 폴더 | 점수 파일 (원시 행) | 본문 후보 → 채택 |",
            "|---|---|---:|"]
    for item in data["sources"]:
        source = item["source"]
        refs = ", ".join(f"{link(source + '/' + f['path'])} ({f['rows']})"
                         for f in item["csv_files"]) or "점수 CSV 없음"
        if item["export_manifest"]:
            refs += f", {link(item['export_manifest'])}"
        out.append(f"| {source} | {refs} | {item['main_candidates']} → {item['main_selected']} |")
    out += ["", "Gaussian Q+BC–FR와 deterministic Q+BC–W2는 본문에서 제외합니다. K=4는 다른 머신에서 실행돼도 같은 방법의 K=4로 합칩니다.", ""]
    if data["main_parser_warnings"]:
        out += ["## 파서 확인 필요", ""] + [f"- {w}" for w in data["main_parser_warnings"]] + [""]
    return "\n".join(out)


def build(root):
    data = inventory(root)
    path = root / "sweep_results/LOG_INDEX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data))
    report = root / "reports/log_index.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"LOG_INDEX.md: {len(data['sources'])} sources, "
          f"{len(data['main_parser_warnings'])} parser warnings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    build(parser.parse_args().root)
