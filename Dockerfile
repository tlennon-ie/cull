# cull — headless/server container image.
#
# Builds a self-contained image that runs the integrated launcher (dashboard +
# pipeline supervisor). The dashboard binds 0.0.0.0:5000 INSIDE the container
# (see pipeline_code/dashboard_enhanced.py: app.run(host="0.0.0.0", ...)), so it
# is reachable from the host once you publish the port (e.g. -p 5000:5000).
#
# Data lives on a mounted volume at /data (PIPELINE_BASE_DIR). Secrets are NOT
# baked in — pass them at run time via `--env-file .env` or docker-compose's
# env_file. See .dockerignore (it keeps .env, data/, and .venv out of the image).
#
#   docker build -t cull .
#   docker run --rm -p 5000:5000 --env-file .env -v "$(pwd)/data:/data" cull
#
# Python 3.12-slim: the project targets Python 3.10+ (union-type syntax in the
# codebase + requirements.txt note). 3.12 is current, slim keeps the image small.
FROM python:3.12-slim

# Faster, quieter, log-friendly Python in a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Where cull keeps queue/, sorted/, logs/, jobs/ — paths.base_dir() reads this.
# Mount a host directory here so curated data survives container recreation.
ENV PIPELINE_BASE_DIR=/data

WORKDIR /app

# ── OS dependencies ──────────────────────────────────────────────────────────
#   ffmpeg        — frame extraction for the upcoming video lane.
#   git           — requirements.txt installs gallery-dl from a git+https URL.
#   build-essential — compile any wheels (e.g. Pillow) that lack a manylinux build.
#   libjpeg/zlib/png/webp + freetype/lcms — Pillow runtime image-format support.
# Cleaned up in the same layer to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libjpeg62-turbo-dev \
        zlib1g-dev \
        libpng-dev \
        libwebp-dev \
        libfreetype6-dev \
        liblcms2-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (own layer for build-cache reuse) ────────────────────
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
# Copy only what the runtime needs (the rest is excluded via .dockerignore).
COPY pipeline_code/ ./pipeline_code/
COPY tools/ ./tools/
COPY launch.sh launch.bat ./

# The mount point for the data volume. Declaring it documents intent and gives
# sane behaviour even when the operator forgets to bind a host directory.
VOLUME ["/data"]

# Dashboard port — bound to 0.0.0.0 inside the container, publish on the host.
EXPOSE 5000

# Default command: the REAL entrypoint (launch.sh ends with
# `exec "$PY" .../integrated_launcher.py`). Boots the dashboard, which owns the
# pipeline Start/Stop lifecycle.
CMD ["python", "-u", "pipeline_code/integrated_launcher.py"]
