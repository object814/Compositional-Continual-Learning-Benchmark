#!/bin/bash
# Build the cclb image, or work in a long-lived container with the repository
# mounted at /workspace.
#
#   bash docker/run.sh build              # build the image
#   bash docker/run.sh                    # open a shell (starts the container if needed)
#   bash docker/run.sh COMMAND [ARGS...]  # run one command in the container
#   bash docker/run.sh stop               # stop and remove the container
#
# The container keeps running after you leave a shell, so tmux sessions and
# runs started inside it continue; run the script again for a new shell.
# Inside you are a user with your host name and IDs and passwordless sudo.
# W&B settings are read from .env when the container starts. Settings:
# CCLB_IMAGE (image, default cclb), CCLB_CONTAINER (container name, default
# cclb-<user>), CCLB_SHM_SIZE (shared memory, default 16g).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CCLB_IMAGE:-cclb}"
NAME="${CCLB_CONTAINER:-cclb-$(id -un)}"
USER_ENV=(-e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e HOST_USER="$(id -un)")

case "${1:-}" in
    build)
        exec docker build -t "$IMAGE" "$REPO_ROOT"
        ;;
    stop)
        if docker rm -f "$NAME" >/dev/null 2>&1; then
            echo "Stopped and removed container $NAME."
        else
            echo "No container named $NAME."
        fi
        exit 0
        ;;
esac

running="$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)"
if [[ -z "$running" ]]; then
    ARGS=(run -d --init --name "$NAME" --gpus all --shm-size "${CCLB_SHM_SIZE:-16g}"
          -v "$REPO_ROOT:/workspace" -w /workspace "${USER_ENV[@]}")
    [[ -f "$REPO_ROOT/.env" ]] && ARGS+=(--env-file "$REPO_ROOT/.env")
    docker "${ARGS[@]}" "$IMAGE" sleep infinity >/dev/null
    echo "Started container $NAME. It keeps running until: bash docker/run.sh stop"
elif [[ "$running" != true ]]; then
    docker start "$NAME" >/dev/null
fi

if [[ "$(docker inspect -f '{{.Image}}' "$NAME")" != "$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" ]]; then
    echo "Note: $NAME runs an older image. To use the new one: bash docker/run.sh stop" >&2
fi

EXEC=(exec -i "${USER_ENV[@]}")
[[ -t 0 && -t 1 ]] && EXEC+=(-t)
[[ $# -gt 0 ]] || set -- bash
exec docker "${EXEC[@]}" "$NAME" cclb-entrypoint "$@"
