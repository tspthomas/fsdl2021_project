FROM continuumio/miniconda3:latest

RUN apt update -y
RUN apt upgrade -y

RUN pip install mlflow==1.14.1
RUN pip install psycopg2-binary
RUN pip install pymysql

# Goes back to workspace
WORKDIR "/workspace"
