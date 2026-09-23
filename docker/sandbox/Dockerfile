FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sandbox

USER sandbox
WORKDIR /tmp
