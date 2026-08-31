#!/usr/bin/env python3
"""Create a deterministic, anonymous AAAI-26 code-and-data bundle."""
from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path

GPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GPT_ROOT.parent
OUTPUT = GPT_ROOT / "gpt_aaai26_bundle.zip"
FIXED_TIMESTAMP = (2026, 8, 31, 0, 0, 0)

GPT_FILES = (
    "paper.tex",
    "supplement.tex",
    "ReproducibilityChecklist.tex",
    "aaai2026.sty",
    "references.bib",
    "build_pdf.sh",
    "verify_bundle.py",
    "make_figures.py",
    "package_bundle.py",
    "BUILD.md",
    "AUTHOR_KIT_SOURCE.md",
    "README.md",
    "CRITICAL_REVIEW.md",
    "REVISION_NOTES.md",
    "SOURCE_MANIFEST.md",
    "BUNDLE_MANIFEST.txt",
    "data/aggregate_summary.csv",
    "data/budget_means.csv",
    "data/environment_high_means.csv",
    "data/task_cluster_uncertainty.csv",
    "data/audit_claims.json",
    "data/config_summary.csv",
    "figures/architecture.pdf",
    "figures/architecture.png",
    "figures/stability_summary.pdf",
    "figures/stability_summary.png",
    "figures/diagnostics_summary.pdf",
    "figures/diagnostics_summary.png",
    "figures/movement_audit.pdf",
    "figures/movement_audit.png",
    "figures/failure_signature.pdf",
    "figures/failure_signature.png",
    "gpt_aaai26_manuscript.pdf",
    "gpt_aaai26_supplement.pdf",
)

REPO_FILES = (
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "train_td3bc.py",
    "d4rl_data.py",
    "tau_grids.py",
    "launch_mpi_sweep.py",
    "sweep_results/diagnostics/README.md",
    "sweep_results/diagnostics/audit_pack/target_exposure_run.csv",
    "sweep_results/diagnostics/audit_pack/route_intervention_run.csv",
    "sweep_results/diagnostics/audit_pack/simulator_checkpoint.csv",
    "sweep_results/diagnostics/audit_pack/failure_diagnostics_run.csv",
)

SANITIZED_CHECKPOINT_CSVS = {
    "sweep_results/diagnostics/audit_pack/target_exposure_run.csv",
    "sweep_results/diagnostics/audit_pack/simulator_checkpoint.csv",
    "sweep_results/diagnostics/audit_pack/failure_diagnostics_run.csv",
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".tex", ".txt", ".toml"}

SCORE_SOURCES = {
    ("K=1", "Imp"): (0, 1, 2, 3),
    ("K=2", "Imp"): (0, 1, 2, 3),
    ("K=3", "Imp"): (0, 1, 2, 3),
    ("K=4", "Imp"): (0, 1, 2, 3),
    ("K=2", "Exp"): (0, 1),
    ("K=3", "Exp"): (0, 1),
}


def archive_files() -> list[Path]:
    files = [GPT_ROOT / rel for rel in GPT_FILES]
    files.extend(REPO_ROOT / rel for rel in REPO_FILES)
    for (depth, realization), seeds in SCORE_SOURCES.items():
        files.extend(
            REPO_ROOT / "sweep_results" / depth / realization / f"seed{seed}.csv"
            for seed in seeds
        )
    for folder in (REPO_ROOT / "scripts" / "diagnostics", REPO_ROOT / "tests"):
        files.extend(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh", ".md"}
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing bundle inputs: " + ", ".join(map(str, missing)))
    unique = {path.resolve(): path for path in files}
    return sorted(unique.values(), key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def archive_payload(path: Path) -> bytes:
    name = path.relative_to(REPO_ROOT).as_posix()
    if name in SANITIZED_CHECKPOINT_CSVS:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        if not rows or "checkpoint" not in rows[0]:
            raise ValueError(f"expected checkpoint column in {name}")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            checkpoint_name = Path(row["checkpoint"]).name
            identity = "/".join(
                f"{key}={row[key]}" for key in ("method", "env", "T", "seed") if key in row)
            row["checkpoint"] = f"artifact://checkpoint/{identity}/{checkpoint_name}"
            writer.writerow(row)
        data = output.getvalue().encode("utf-8")
    else:
        data = path.read_bytes()
    sensitive_markers = (b"/" + b"home/", b"ext" + b"_csv")
    if path.suffix in TEXT_SUFFIXES and any(marker in data for marker in sensitive_markers):
        raise ValueError(f"local workspace identifier found in archive member {name}")
    return data


def write_bundle() -> tuple[int, int]:
    files = archive_files()
    temporary = OUTPUT.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.comment = b"Anonymous AAAI-26 BAR manuscript, code, compact data, and verification bundle"
            for path in files:
                name = path.relative_to(REPO_ROOT).as_posix()
                info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if os.access(path, os.X_OK) else 0o644
                info.external_attr = mode << 16
                archive.writestr(
                    info,
                    archive_payload(path),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    return len(files), OUTPUT.stat().st_size


if __name__ == "__main__":
    count, size = write_bundle()
    print(f"built: {OUTPUT} ({count} files, {size} bytes)")
