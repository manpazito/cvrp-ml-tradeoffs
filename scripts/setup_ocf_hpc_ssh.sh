#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="hpcctl.ocf.berkeley.edu"
HPC_ALIAS="berkeley-hpc"
HPC_USER_DEFAULT="manpazito"
SSH_DIR="${HOME}/.ssh"
SSH_CONFIG="${SSH_DIR}/config"
KEY_PATH="${SSH_DIR}/id_ed25519_ocf"

HPC_USER="${HPC_USER:-${HPC_USER_DEFAULT}}"

if ! command -v ssh >/dev/null 2>&1; then
  echo "Error: ssh is not installed or not in PATH." >&2
  exit 1
fi

mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"

touch "${SSH_CONFIG}"
chmod 600 "${SSH_CONFIG}"

if [[ ! -f "${KEY_PATH}" ]]; then
  echo "Creating SSH key at ${KEY_PATH}"
  ssh-keygen -t ed25519 -f "${KEY_PATH}" -C "${USER}@${HOSTNAME}-ocf-hpc" -N ""
else
  echo "SSH key already exists: ${KEY_PATH}"
fi

if grep -qE "^Host[[:space:]]+${HPC_ALIAS}([[:space:]]|$)" "${SSH_CONFIG}"; then
  echo "SSH config entry already exists for host alias '${HPC_ALIAS}'."
else
  {
    echo ""
    echo "# Berkeley OCF HPC (added by scripts/setup_ocf_hpc_ssh.sh)"
    echo "Host ${HPC_ALIAS} ${HPC_HOST}"
    echo "  HostName ${HPC_HOST}"
    echo "  User ${HPC_USER}"
    echo "  IdentityFile ${KEY_PATH}"
    echo "  IdentitiesOnly yes"
    echo "  ServerAliveInterval 60"
    echo "  ServerAliveCountMax 120"
    echo "  ControlMaster auto"
    echo "  ControlPath ~/.ssh/cm-%r@%h:%p"
    echo "  ControlPersist 10m"
  } >> "${SSH_CONFIG}"
  echo "Added SSH config entry for '${HPC_ALIAS}' in ${SSH_CONFIG}."
fi

echo ""
echo "Next steps:"
echo "1) Install your public key on OCF HPC (run once):"
if command -v ssh-copy-id >/dev/null 2>&1; then
  echo "   ssh-copy-id -i ${KEY_PATH}.pub ${HPC_USER}@${HPC_HOST}"
else
  echo "   cat ${KEY_PATH}.pub"
  echo "   # then add it to ~/.ssh/authorized_keys on the remote account"
fi
echo "2) Connect to Slurm control/login node:"
echo "   ssh ${HPC_ALIAS}"
echo "   # equivalent direct command: ssh ${HPC_USER}@${HPC_HOST}"
echo "3) Quick Slurm check after login:"
echo "   sinfo"
