# MLFlow Tracking Server
Simple setup with MLFlow Tracking Server plus Postgres as the storage backend.

## Setup

### Creating environment variables

Create a file called .env containing the environment variables for all services.

```
touch .env
```

Add the following variables.
```bash
POSTGRES_DB=mlflow
POSTGRES_DATA=<directory to persist postgress stuff>
POSTGRES_USER=<username>
POSTGRES_PASSWORD=<password>
POSTGRES_SERVICE_NAME=postgres

MLFLOW_SERVER_PORT=5033
MLFLOW_ARTIFACT_ROOT=<directory to persist mlflow artifacts>
```

**IMPORTANT:** Do not version this file in git.

### Starting the System
```
docker-compose build
docker-compose up
```

You should be able to open MLFlow UI in `http://<host_ip>:5033` (or the port you specified in `MLFLOW_SERVER_PORT`).
