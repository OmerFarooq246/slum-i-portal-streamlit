FROM --platform=linux/arm64 mambaorg/micromamba:2.9.0-debian13-slim

ENV CUDA_VISIBLE_DEVICES=-1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=2

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER benchmarks/environment.yml /tmp/environment.yml
RUN micromamba install --yes --name base --file /tmp/environment.yml \
    && micromamba clean --all --yes

COPY config /app/config
COPY ml /app/ml
COPY benchmarks/run_enb5_system_benchmark.py /app/benchmarks/run_enb5_system_benchmark.py

ENTRYPOINT ["micromamba", "run", "--no-capture-output", "--name", "base", "python", "benchmarks/run_enb5_system_benchmark.py"]
