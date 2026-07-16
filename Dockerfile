FROM python:3.12-slim

WORKDIR /app

# Install the package first so the layer caches when only docs change.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docs ./docs

ENTRYPOINT ["agentic-rag"]
CMD ["ask", "What are the core working hours?"]
