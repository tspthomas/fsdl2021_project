#!/bin/bash
mlflow ui \
    --host 0.0.0.0 \
    --port $MLFLOW_SERVER_PORT \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT