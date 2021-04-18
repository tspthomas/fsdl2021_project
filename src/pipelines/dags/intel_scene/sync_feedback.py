import os
import numpy as np
import pandas as pd
import pickle

import mlflow

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
np.random.seed(33)

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='sync_feedback',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'dataset']
) as dag:

    sync_dataset_task = BashOperator(
        task_id='sync_feedback',
        bash_command='./sync_feedback.sh',
        dag=dag,
    )

    sync_dataset_task

if __name__ == "__main__":
    dag.cli()