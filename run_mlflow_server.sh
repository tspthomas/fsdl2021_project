#!/bin/bash
mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_SERVER_PORT \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT