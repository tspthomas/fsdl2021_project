FROM apache/airflow:2.0.1

USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt