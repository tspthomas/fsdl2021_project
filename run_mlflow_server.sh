#!/bin/bash
mlflow ui \
    --host 0.0.0.0 \
    --port $MLFLOW_SERVER_PORT \
    --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_SERVICE_NAME/$POSTGRES_DB \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT