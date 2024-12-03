FROM python:3.8
ENV PYTHONUNBUFFERED 1

# Allows docker to cache installed dependencies between builds
#COPY ./requirements/base.txt ./requirements/local.txt ./

COPY . /code
WORKDIR /code

RUN pip install -r ./requirements.txt

RUN apt-get update
RUN apt-get install gettext -y
RUN django-admin compilemessages -l en
#RUN django-admin compilemessages -l ar

# Adds our application code to the image

ENV WAIT_VERSION 2.7.2
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/$WAIT_VERSION/wait /wait
RUN chmod +x /wait


EXPOSE 8587
