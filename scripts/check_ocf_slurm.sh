#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/connect_ocf_hpc.sh" "hostname; whoami; command -v sinfo >/dev/null 2>&1 && sinfo || scontrol ping"
