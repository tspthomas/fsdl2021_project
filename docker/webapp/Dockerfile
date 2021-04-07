FROM continuumio/miniconda3:latest

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y git

COPY docker/webapp/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt