FROM apache/airflow:2.0.1

USER root
RUN apt update -y
RUN apt upgrade -y
RUN apt install -y git

# Install custom Python lib
COPY src/fsdllib /tmp/fsdllib
RUN cd /tmp/fsdllib && python setup.py install

USER airflow
ARG AIRFLOW_HOME=/opt/airflow

COPY docker/airflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# https://github.com/apache/airflow/issues/14266
RUN pip install apache-airflow-providers-microsoft-azure==1.2.0rc1 