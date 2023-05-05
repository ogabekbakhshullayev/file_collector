FROM python:3.11.3

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./app/ ./

COPY ./images/ /images/

ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080
