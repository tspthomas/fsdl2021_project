#!/bin/bash

YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${NC}Initializing Airflow environment"
echo -e "--------------------------------\n"

echo -e "${NC}[Info] Creating required directories"
if [[ -d ./src/pipelines/logs ]]
then
    echo -e "${YELLOW}[Warning] ./src/logs already exists"
else
    mkdir ./src/pipelines//logs
    echo -e "${NC}[Info] Created directory ./src/logs"
fi

if [[ -d ./src/pipelines//plugins ]]
then
    echo -e "${YELLOW}[Warning] ./src/plugins already exists"
else
    mkdir ./src/pipelines//plugins
    echo -e "${NC}[Info] Created directory ./src/plugins"
fi

echo -e "${NC}\n[Info] Creating required environment variables"

AIRFLOW_UID=$(grep AIRFLOW_UID ./.env | cut -d '=' -f2)
AIRFLOW_GID=$(grep AIRFLOW_GID ./.env | cut -d '=' -f2)

if [[ -z "$AIRFLOW_UID" ]]
then
    echo -e "\nAIRFLOW_UID=$(id -u)" >> ./.env
    echo -e "${NC}[Info] Added AIRFLOW_UID to .env"
else
    echo -e "${YELLOW}[Warning] AIRFLOW_UID already set in .env file"
fi

if [[ -z "$AIRFLOW_GID" ]]
then
    echo -e "\nAIRFLOW_GID=0" >> ./.env
    echo -e "${NC}[Info] Added AIRFLOW_GID to .env"
else
    echo -e "${YELLOW}[Warning] AIRFLOW_GID already set in .env file"
fi

echo -e "${NC}\n[Info] Done!"