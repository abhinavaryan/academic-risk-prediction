# Production image: builds the static UI, then serves UI + API from one FastAPI process.
# Used for Hugging Face Spaces (port 7860) and any Docker host.
# Base images can be swapped for a mirror, e.g.
#   --build-arg NODE_IMAGE=public.ecr.aws/docker/library/node:22-slim
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${NODE_IMAGE} AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json web/.npmrc ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVE_STATIC=1 \
    PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY riskml ./riskml
COPY artifacts ./artifacts
COPY sample_data ./sample_data
COPY --from=web /app/web/out ./web/out
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
CMD ["sh", "-c", "uvicorn riskml.server.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
