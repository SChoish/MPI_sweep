# offrl `mpi4_norm` import manifest

Checksum / provenance pack for Prox-4 (`results/mpi4_norm`) hop-wise import.

| File | Contents |
| --- | --- |
| `MANIFEST.csv` | 504 runs: sha256, bytes, mtime, config/eval/log presence |
| `params_1000000_sha256.csv` | `relpath,sha256` for every `params_1000000.pkl` |
| `LAUNCH.json` | launch waves, trainer git recovery, matrix, rsync status |
| `MANIFEST_META.json` | pack metadata |

Expected matrix: 9 env × 14 τ × seed `{0,1,2,3}` = **504**. Missing cells: **0**.
