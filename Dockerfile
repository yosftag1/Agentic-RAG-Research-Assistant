# ═══════════════════════════════════════════════════════════════════
# DOCKERFILE — Packages Our App for Deployment
#
# 📚 DOCKER CRASH COURSE:
#
# Docker creates a "container" — like a lightweight virtual machine
# that has everything our app needs to run. Think of it as:
#   "Ship your laptop's Python environment to the cloud"
#
# Each line is a step:
#   FROM   → start with a base image (pre-built OS + Python)
#   WORKDIR → set the working directory (like cd)
#   COPY   → copy files from your machine into the container
#   RUN    → run a command during build (like pip install)
#   EXPOSE → document which port the app uses
#   CMD    → the command to run when the container starts
#
# Build this:  docker build -t research-assistant .
# Run this:    docker run -p 8000:8000 research-assistant
# ═══════════════════════════════════════════════════════════════════

# Start with Python 3.12 on a slim Debian base (small image size)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first (Docker caches this layer)
# If dependencies don't change, Docker won't re-install them
COPY pyproject.toml README.md ./

# Copy the source code
COPY src/ ./src/

# Install the package in non-editable mode
# --no-cache-dir saves space by not caching pip downloads
RUN pip install --no-cache-dir .

# Create data and chroma_db directories
RUN mkdir -p data chroma_db

# Expose port 8000 (documentation for humans, doesn't actually open it)
EXPOSE 8000

# Run the server when the container starts
# $PORT is set by Render/Railway — defaults to 8000 for local use
CMD uvicorn research_assistant.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
