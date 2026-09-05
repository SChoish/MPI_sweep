#!/usr/bin/env python3
"""Write the P0 K=4 hop-actor checkpoint inventory. No training."""
from __future__ import annotations

import json
from pathlib import Path

from _p0_hop_common import ARCHIVE, write_inventory


def main() -> int:
    inventory = write_inventory()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    print(json.dumps(inventory["coverage"], indent=2), flush=True)
    print(f"wrote {ARCHIVE / 'INVENTORY.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
