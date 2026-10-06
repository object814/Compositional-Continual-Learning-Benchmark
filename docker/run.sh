#!/bin/bash
# Build or run the cclb Docker image with the repository mounted at /workspace.
#
#   bash docker/run.sh build              # build the image
#   bash docker/run.sh                    # interactive shell
#   bash docker/run.sh COMMAND [ARGS...]  # run one command, e.g. python scripts/check_env.py
#
# Inside the container you are a user with your host name and IDs and
# passwordless sudo. W&B settings are read from .env at the repository root
# (copy .env.example). CCLB_IMAGE sets the image tag (default cclb) and
# CCLB_SHM_SIZE the shared-memory size (default 16g).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CCLB_IMAGE:-cclb}"

if [[ "${1:-}" == build ]]; then
    exec docker build -t "$IMAGE" "$REPO_ROOT"
fi

ARGS=(run --rm -i --gpus all --shm-size "${CCLB_SHM_SIZE:-16g}"
      -v "$REPO_ROOT:/workspace" -w /workspace
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e HOST_USER="$(id -un)")
[[ -t 0 && -t 1 ]] && ARGS+=(-t)
[[ -f "$REPO_ROOT/.env" ]] && ARGS+=(--env-file "$REPO_ROOT/.env")
[[ $# -gt 0 ]] || set -- bash

exec docker "${ARGS[@]}" "$IMAGE" "$@"
