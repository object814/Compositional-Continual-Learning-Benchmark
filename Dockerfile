# Environment for all baselines; the repository is mounted at /workspace at run time.
# Build and use with docker/run.sh:
#   bash docker/run.sh build
#   bash docker/run.sh          # shell in a long-lived container
#   bash docker/run.sh stop
#
# Debian 12 provides a C++ runtime (libstdc++) new enough for the compiled
# tensordict extension; Python packages come from requirements.txt.
FROM python:3.10-slim-bookworm

# libOSMesa for offscreen MuJoCo rendering, OpenGL/GLib runtime libraries,
# sudo, and tmux for sessions that keep running inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libegl1 libglib2.0-0 sudo tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/cclb-env/requirements.txt
RUN PYTHONNOUSERSITE=1 pip install --no-cache-dir -r /opt/cclb-env/requirements.txt

# Runs commands as a user matching the host user (see docker/entrypoint.sh).
COPY docker/entrypoint.sh /usr/local/bin/cclb-entrypoint
RUN chmod 755 /usr/local/bin/cclb-entrypoint

# Same variables as environment.yml. HOME=/tmp is a writable fallback; the
# entrypoint sets the user's home directory.
ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /workspace
ENTRYPOINT ["cclb-entrypoint"]
CMD ["bash"]
