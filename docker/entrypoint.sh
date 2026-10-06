#!/bin/bash
# Container entrypoint, also used by docker/run.sh for every shell it opens:
# when HOST_UID/HOST_GID/HOST_USER are set, run the command as a matching user
# with passwordless sudo, so files written to the mounted repository belong to
# the host user. Safe to run repeatedly and concurrently.
set -e

if [[ "$(id -u)" == 0 && -n "${HOST_UID:-}" && "$HOST_UID" != 0 ]]; then
    gid="${HOST_GID:-$HOST_UID}"
    name="${HOST_USER:-cclb}"
    getent group "$gid" >/dev/null || groupadd -g "$gid" "$name" 2>/dev/null || true
    if ! getent passwd "$HOST_UID" >/dev/null; then
        id "$name" >/dev/null 2>&1 && name=cclb
        useradd -u "$HOST_UID" -g "$gid" -m -s /bin/bash "$name" 2>/dev/null || true
    fi
    name="$(getent passwd "$HOST_UID" | cut -d: -f1)"
    [[ -n "$name" ]] || { echo "Could not create a user with ID $HOST_UID" >&2; exit 1; }
    echo "$name ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/cclb
    chmod 440 /etc/sudoers.d/cclb
    export USER="$name" HOME="$(getent passwd "$HOST_UID" | cut -d: -f6)"
    exec setpriv --reuid="$HOST_UID" --regid="$gid" --init-groups "$@"
fi

exec "$@"
