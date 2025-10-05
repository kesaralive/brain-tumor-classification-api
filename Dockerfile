# Stage 1: Build / Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy Poetry files first
COPY pyproject.toml poetry.lock* /app/


# Copy app source
COPY . .

# Install app itself (also only main)
RUN poetry install 


EXPOSE 8081

# Run the FastAPI server
CMD ["poetry", "run", "server"]