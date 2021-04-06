# FSDL 2021 Project

## Setup

### Creating environment variables

Create a file called .env containing the environment variables for all services.

```
touch .env
```

Add the following variables.
```bash
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

MLFLOW_SERVER_PORT=5033
MLFLOW_ARTIFACT_ROOT=/data/experiments/
MLFLOW_TRACKING_URI=http://mlflow:5033

AIRFLOW_PORT=8080
AIRFLOW_UID=1000
AIRFLOW_GID=0
```

**IMPORTANT:** Do not version this file in git.

### Starting the System
```
docker-compose build
docker-compose up
```

You should be able to open MLFlow UI in `http://<host_ip>:5033` (or the port you specified in `MLFLOW_SERVER_PORT`).

You should be able to open AIrflow UI in `http://<host_ip>:8080` (or the port you specified in `AIRFLOW_PORT`).