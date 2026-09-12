# Runtime configuration pins the resulting image ID/digest, not this build tag.
FROM python:3.12-slim
WORKDIR /app
COPY deploy/solver/requirements.txt /build/requirements.txt
RUN python -m pip install --no-cache-dir -r /build/requirements.txt
COPY core/*.py /app/core/
COPY utils/*.py /app/utils/
RUN python -c "import scipy.integrate, scipy.optimize, scipy.stats; import core.solver_worker; import core.mechanistic_modeling; import core.graph_confirmation"
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "-u", "/app/core/solver_worker.py"]
