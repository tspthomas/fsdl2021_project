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
MLFLOW_ARTIFACT_ROOT=./mlruns/
MLFLOW_BACKEND_STORE_URI=sqlite:///database/mlflow.db
MLFLOW_TRACKING_URI=http://mlflow:${MLFLOW_SERVER_PORT}

AIRFLOW_PORT=8081
FLOWER_PORT=5556
```

**IMPORTANT:** Do not version this file in git.

### Creating Airflow environment variables

Simply run the following command in the root directory of this repository
```
./airflow/init_env.sh
```

Check the environment variables `AIRFLOW_UID` and `AIRFLOW_GID` were created in the `.env` file. Also, double-check the directories `src/logs` and `src/plugins` were created.

### Starting the System

Build Docker images
```
docker-compose build
```

Start containers
```
docker-compose up
```

In case you want to stop
```
docker-compose down
```

### Accessing Airflow UI
You should be able to open AIrflow UI in `http://<host_ip>:8080` (or the port you specified in `AIRFLOW_PORT`).

### Accessing MLFlow UI
You should be able to open MLFlow UI in `http://<host_ip>:5033` (or the port you specified in `MLFLOW_SERVER_PORT`).