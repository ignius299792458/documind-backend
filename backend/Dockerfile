# ============================================================
# Dockerfile — DocuMind FastAPI Backend
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install dependencies with Poetry
#   Stage 2 (runtime): copy only what's needed to run
#
# Multi-stage keeps the final image small by leaving Poetry,
# pip cache, and build tools out of the production image.
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first (Docker layer cache: only re-installs
# when pyproject.toml or poetry.lock changes, not on code changes)
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt, then install with pip.
# This avoids having Poetry in the final image.
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security — don't run as root in production
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code (lives under src/ in the repo)
COPY src/documind_backend/ ./documind_backend/

# Create uploads directory with correct ownership
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server: Uvicorn with 2 workers
# Adjust --workers based on CPU cores (formula: 2 * CPUs + 1)
CMD ["uvicorn", "documind_backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]