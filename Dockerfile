FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAP_HOME=/data

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home app && mkdir /data && chown app /data
USER app
VOLUME /data

ENTRYPOINT ["faketrap"]
CMD ["--help"]
