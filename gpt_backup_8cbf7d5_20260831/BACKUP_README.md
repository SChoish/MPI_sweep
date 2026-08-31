# Snapshot of revised GPT AAAI-26 manuscript

Taken **2026-08-31 15:46 KST** before a later GPT pass overwrites `gpt/`.

- Git commit: `8cbf7d5` (`Revise GPT AAAI-26 BAR manuscript toward exposure-vs-reach.`)
- Title at snapshot: BAR decouples bootstrap exposure from policy reach
- Do not let GPT edit this folder. Restore with:

```bash
rsync -a --delete gpt_backup_8cbf7d5_20260831/ gpt/ --exclude BACKUP_README.md
```

A second copy lives at `/home/ext_csv/backups/gpt_backup_8cbf7d5_20260831/`.
