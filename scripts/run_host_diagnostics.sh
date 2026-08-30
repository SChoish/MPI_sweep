#!/usr/bin/env bash
# Compatibility shim — prefer scripts/diagnostics/run_host.sh
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/diagnostics/run_host.sh" "$@"
