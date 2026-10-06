#!/bin/bash
# Container entrypoint: when started by docker/run.sh (HOST_UID/HOST_GID/HOST_USER
# set), run the command as a matching user with passwordless sudo, so files
# written to the mounted repository belong to the host user.
set -e

if [[ "$(id -u)" == 0 && -n "${HOST_UID:-}" && "$HOST_UID" != 0 ]]; then
    gid="${HOST_GID:-$HOST_UID}"
    name="${HOST_USER:-cclb}"
    getent group "$gid" >/dev/null || groupadd -g "$gid" "$name"
    if ! getent passwd "$HOST_UID" >/dev/null; then
        id "$name" >/dev/null 2>&1 && name=cclb
        useradd -u "$HOST_UID" -g "$gid" -m -s /bin/bash "$name"
    fi
    name="$(getent passwd "$HOST_UID" | cut -d: -f1)"
    echo "$name ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/cclb
    chmod 440 /etc/sudoers.d/cclb
    export USER="$name" HOME="$(getent passwd "$HOST_UID" | cut -d: -f6)"
    exec setpriv --reuid="$HOST_UID" --regid="$gid" --init-groups "$@"
fi

exec "$@"
