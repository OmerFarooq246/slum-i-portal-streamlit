FROM --platform=linux/arm64 python:3.11-slim-bookworm

ENV CUDA_VISIBLE_DEVICES=-1 \
    KERAS_HOME=/root/.keras \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=2

WORKDIR /app

COPY benchmarks/requirements.txt /tmp/benchmark-requirements.txt
RUN pip install --no-cache-dir -r /tmp/benchmark-requirements.txt

COPY config /app/config
COPY ml /app/ml
COPY benchmarks/run_enb5_system_benchmark.py /app/benchmarks/run_enb5_system_benchmark.py

ENTRYPOINT ["python", "benchmarks/run_enb5_system_benchmark.py"]
