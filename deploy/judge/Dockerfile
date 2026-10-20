FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY deploy/judge/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY --chown=user core /app/core
COPY --chown=user apps /app/apps
COPY --chown=user bench/faults /app/bench/faults
COPY --chown=user jobs/sandbox /app/jobs/sandbox
COPY --chown=user LICENSE /app/LICENSE
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CMD ["uvicorn", "apps.judge.service:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
