FROM python:3.11.6-slim as mc
ENV I_AM_INSIDE_DOCKER_CONTAINER=true
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    make \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements /app/requirements
RUN pip install --upgrade pip
RUN pip install --upgrade --no-cache-dir -r requirements/dev.txt
CMD sleep infinity
