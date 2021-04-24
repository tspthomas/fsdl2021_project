FROM continuumio/miniconda3:latest

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y git

COPY docker/webapp/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Pre load feature extractors
COPY docker/webapp/load_feature_extractors.py /tmp/load_feature_extractors.py
RUN python /tmp/load_feature_extractors.py

# Install custom Python lib
COPY src/fsdllib /tmp/fsdllib
RUN cd /tmp/fsdllib && python setup.py install