# Environment for all baselines. The image holds only the Python environment;
# the repository is mounted at /workspace when the container starts (see README).
#
# Debian 12 provides a C++ runtime (libstdc++) new enough for the compiled
# tensordict extension; Python packages come from requirements.txt.
FROM python:3.10-slim-bookworm

# libOSMesa for offscreen MuJoCo rendering, OpenGL/GLib runtime libraries,
# and tmux for sessions that keep running inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libegl1 libglib2.0-0 tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Same rendering variables as environment.yml. The container runs as the host
# user (docker run --user, or user: in compose.yaml), who has no entry in /etc/passwd: USER gives
# getpass.getuser() (used by torch.compile) a name, and HOME a writable directory.
ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USER=cclb \
    HOME=/tmp \
    SHELL=/bin/bash
WORKDIR /workspace

# Keep the container alive in the background; open shells with docker exec.
CMD ["sleep", "infinity"]
