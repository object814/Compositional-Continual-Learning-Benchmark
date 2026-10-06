# Environment for all baselines; the repository is mounted at /workspace at run time.
#   docker build -t cclb .
#   docker run --rm -it --gpus all -u "$(id -u):$(id -g)" -v "$PWD":/workspace cclb
FROM condaforge/miniforge3:24.11.3-0

# libOSMesa for offscreen MuJoCo rendering, plus OpenGL/GLib runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml requirements.txt /opt/cclb-env/
RUN PYTHONNOUSERSITE=1 conda env create -p /opt/cclb -f /opt/cclb-env/environment.yml \
    && conda clean -afy \
    && rm -rf /root/.cache/pip

# Same variables as environment.yml; HOME=/tmp keeps tool caches writable for any user ID.
ENV PATH=/opt/cclb/bin:$PATH \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /workspace
CMD ["bash"]
