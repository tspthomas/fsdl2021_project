#!/bin/bash

YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${NC}Cleaning up MLFlow data"
echo -e "--------------------------------\n"

echo -e "${NC}[Info] Deleting database"
if [[ -f ./storage/database/mlflow.db ]]
then
    rm -rf ./storage/database/mlflow.db
    echo -e "${NC}[Info] Deleted database in ./storage/database/mlflow.db"
else
    echo -e "${YELLOW}[Warning] ./storage/database/mlflow.db doesn't exist"
fi

echo -e "\n${NC}[Info] Deleting artifacts"
if [[ -d ./storage/artifacts/ ]]
then
    sudo rm -rf ./storage/artifacts/*
    echo -e "${NC}[Info] Deleted all artifacts inside ./storage/artifacts"
else
    echo -e "${YELLOW}[Warning] ./storage/artifacts doesn't exist"
fi

echo -e "${NC}\n[Info] Done!"