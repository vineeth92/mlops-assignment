FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install scikit-learn joblib

CMD ["python", "predict.py"]

