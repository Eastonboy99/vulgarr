FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    NUDENET_CACHE=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
  && pip3 install --no-cache-dir -r requirements.txt \
  # nudenet depends on plain onnxruntime (CPU), whose files collide with
  # onnxruntime-gpu's (same module name) -- reinstall the GPU wheel last,
  # without deps, so its CUDA build wins.
  && pip3 install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.20.2

COPY server.py .

VOLUME ["/models"]
EXPOSE 7861

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7861"]
