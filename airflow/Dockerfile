FROM apache/airflow:2.0.1

USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# https://github.com/apache/airflow/issues/14266
RUN pip install apache-airflow-providers-microsoft-azure==1.2.0rc1 