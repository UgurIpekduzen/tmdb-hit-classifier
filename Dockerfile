FROM python:3.10-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.135.1 \
    uvicorn[standard]==0.40.0 \
    mlflow==3.11.1 \
    pandas==2.2.2 \
    numpy==2.0.1 \
    scipy==1.15.3 \
    lightgbm==4.6.0 \
    scikit-learn==1.7.2 \
    pydantic==2.12.5

COPY app/ app/
COPY data/ data/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_STAGE=Production
ENV DATASET_PATH=data/tmdb_model.csv

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
