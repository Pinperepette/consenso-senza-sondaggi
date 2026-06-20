FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONSENSO_MONGO_URI=mongodb://mongo:27017 \
    CONSENSO_DB=consenso \
    FLASK_APP=consenso.api.app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x docker/entrypoint.sh

EXPOSE 5057
ENTRYPOINT ["/app/docker/entrypoint.sh"]
# 1 worker (JAX), thread per le richieste; timeout alto per sync/inferenza
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-t", "600", \
     "-b", "0.0.0.0:5057", "consenso.api.app:app"]
