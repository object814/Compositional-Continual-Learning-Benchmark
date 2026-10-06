# Environment for all baselines; the repository is mounted at /workspace at run time.
# Build and use with docker/run.sh:
#   bash docker/run.sh build
#   bash docker/run.sh          # shell in a long-lived container
#   bash docker/run.sh stop
FROM condaforge/miniforge3:24.11.3-0

# libOSMesa for offscreen MuJoCo rendering, OpenGL/GLib runtime libraries, and sudo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 libgl1 libegl1 libglib2.0-0 sudo \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml requirements.txt /opt/cclb-env/
RUN PYTHONNOUSERSITE=1 conda env create -p /opt/cclb -f /opt/cclb-env/environment.yml \
    && conda clean -afy \
    && rm -rf /root/.cache/pip

# Runs the command as a user matching the host user (see docker/entrypoint.sh).
COPY docker/entrypoint.sh /usr/local/bin/cclb-entrypoint
RUN chmod 755 /usr/local/bin/cclb-entrypoint

# The base image's .bashrc activates conda's own "base" environment, which
# would shadow /opt/cclb in interactive shells; activate /opt/cclb instead.
RUN for f in /etc/skel/.bashrc /root/.bashrc; do \
        sed -i 's#conda activate base#conda activate /opt/cclb#' "$f"; \
        grep -q 'conda activate /opt/cclb' "$f" \
            || echo '. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/cclb' >> "$f"; \
    done

# tmux, for sessions that keep running inside the long-lived container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*

# Same variables as environment.yml. HOME=/tmp is a writable fallback; the
# entrypoint sets the user's home directory.
ENV PATH=/opt/cclb/bin:$PATH \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /workspace
ENTRYPOINT ["cclb-entrypoint"]
CMD ["bash"]
