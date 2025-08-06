FROM python:3.10

ENV PYTHONUNBUFFERED 1

RUN apt-get update -y && apt-get install libgl1 -y


WORKDIR /app
COPY ./requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./app/ ./
# Ensure static directory exists for the test interface
RUN mkdir -p ./static

ENV PYTHONPATH "/app"

EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080
