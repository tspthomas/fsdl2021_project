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
MLFLOW_ARTIFACT_ROOT=./storage/artifacts/
MLFLOW_BACKEND_STORE_URI=sqlite:///storage/database/mlflow.db
MLFLOW_TRACKING_URI=http://mlflow:${MLFLOW_SERVER_PORT}

AIRFLOW_PORT=8081
FLOWER_PORT=5556

FLASK_PORT=8082
```

**IMPORTANT:** Do not version this file in git.

### Creating Airflow environment variables

Simply run the following command in the root directory of this repository
```
./docker/airflow/init_env.sh
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
You should be able to open Airflow UI in `http://<host_ip>:8081` (or the port you specified in `AIRFLOW_PORT`).

### Accessing MLFlow UI
You should be able to open MLFlow UI in `http://<host_ip>:5033` (or the port you specified in `MLFLOW_SERVER_PORT`).

### Accessing Web App UI
You should be able to open Web App UI in `http://<host_ip>:8082` (or the port you specified in `FLASK_PORT`).

### Cleaning up MLFlow data (optional)

Simply run the following command in the root directory of this repository
```
./docker/mlflow/clean_mlflow_data.sh
```

### DVC
DVC is enabled in `/data/raw` and linked to Google Drive. Steps taken to initiate the dvc repo and link to Google Drive. Commands to initiate: 
```
docker exec -it fsdl2021_project_airflow-worker_1 /bin/bash
cd /workspace/data/raw
dvc init --no-scm
dvc remote add -d storage gdrive://<end_of_google_drive_link>
# authentication required
dvc add /workspace/data/raw/intel_image_scene
dvc push
```

`/workspace/data/raw/.dvc` contains the configuration files necessary for dvc to track data files and directories. 

DVC allows the dataset to be stored on Google Drive. Downloading data:
```
dvc pull
```

DVC allows version control of data. For example, when new files are added:
```
cd /workspace/data/raw
dvc status # show the changes
dvc add intel_image_scene
dvc push
```

At this time, still researching how to checkout previous versions without git dependency.