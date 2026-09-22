FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "[ -f models_bundle.pkl ] || uv run --no-sync python train_automl.py; uv run --no-sync streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
