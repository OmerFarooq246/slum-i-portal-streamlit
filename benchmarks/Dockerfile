FROM --platform=linux/arm64 mambaorg/micromamba:2.9.0-debian13-slim

ENV CUDA_VISIBLE_DEVICES=-1 \
    PYTHONPATH=. \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=2

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER benchmarks/environment.yml benchmark-environment.yml
RUN micromamba install --yes --name base --file benchmark-environment.yml \
    && micromamba clean --all --yes

COPY config config
COPY ml ml
COPY benchmarks/system_benchmark.py benchmarks/system_benchmark.py
RUN micromamba run --no-capture-output --name base \
    python benchmarks/system_benchmark.py --help >/dev/null

ENTRYPOINT ["micromamba", "run", "--no-capture-output", "--name", "base", "python", "benchmarks/system_benchmark.py"]
CMD ["--model", "enb5_seg_islamabad", "--batch-size", "4", "--cpu-limit", "2"]
