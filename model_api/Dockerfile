FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# COPY ./model_api/app /model_api/app
# COPY ./model_package/cnn_model /model_package/cnn_model
# COPY ./requirements.txt /tmp

# WORKDIR /tmp

# RUN pip3 install -r requirements.txt






# FROM ubuntu:19.10

# COPY ./model_api/api /model_api/api

# COPY ./model_package/cnn_model /model_package/cnn_model

# COPY requirements.txt /requirements.txt

# RUN apt-get update \
#     && apt-get install python3-dev python3-pip -y \
#     && pip3 install -r requirements.txt

# ENV PYTHONPATH=/api
# WORKDIR /api

# EXPOSE 8000

# ENTRYPOINT ["uvicorn"]
# CMD ["api.main:app", "--host", "0.0.0.0"]