FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1@sha256:d1c8e971ab994235840eacc31c4ef4173bf9156317b1bf8aabe7e01eb21b2a0e

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ENABLE_VLM=0 \
    ENABLE_GPIO=0 \
    APP_PORT=5050 \
    TCP_PORT=5566 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl libgl1 libglib2.0-0 libgomp1 libopenblas0-pthread python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/streetlight-venv
ENV PATH=/opt/streetlight-venv/bin:$PATH

WORKDIR /opt/streetlight
COPY requirements-docker.txt .
RUN python3 -m pip install --no-cache-dir pip==25.0.1 \
    && python3 -m pip install --no-cache-dir --prefer-binary -r requirements-docker.txt \
    && python3 -m pip check

RUN mkdir -p models \
    && curl --fail --location --retry 3 \
        --output models/yolo26n.pt \
        https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
    && printf '%s  %s\n' \
        9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
        models/yolo26n.pt | sha256sum -c -

COPY app.py devices.json ./
COPY templates/ templates/

EXPOSE 5050 5566
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail --silent --max-time 3 "http://127.0.0.1:$APP_PORT/api/status" -o /dev/null || exit 1

CMD ["python3", "app.py"]
