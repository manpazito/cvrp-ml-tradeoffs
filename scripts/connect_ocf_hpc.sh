#!/usr/bin/env bash
set -euo pipefail

HPC_ALIAS="berkeley-hpc"
HPC_DIRECT="manpazito@hpcctl.ocf.berkeley.edu"

if ssh -G "${HPC_ALIAS}" >/dev/null 2>&1; then
  exec ssh "${HPC_ALIAS}" "$@"
else
  echo "SSH alias '${HPC_ALIAS}' not found in ~/.ssh/config; falling back to direct host." >&2
  exec ssh "${HPC_DIRECT}" "$@"
fi
