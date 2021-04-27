import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib

import mlflow

from datetime import timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from intel_scene.utils import PROCESSED_DATA_DIR, Data

np.random.seed(33)

def load_data(**context):
    print("Loading Data")

    with open(os.path.join(PROCESSED_DATA_DIR, 'intel_image_scene', 'data.pickle'), 'rb') as f:
        data = pickle.load(f)

    print("Splitting Dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(**context):
    print("Train Model")

    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='load_data')
    X_train = task_instance_data[0]
    X_test = task_instance_data[1]
    y_train = task_instance_data[2]
    y_test = task_instance_data[3]

    mlflow_run = mlflow.start_run()
    mlflow.sklearn.autolog(log_models=False)

    max_iter = 10000
    logmodel = LogisticRegression(max_iter=max_iter)
    logmodel.fit(X_train, y_train)

    return logmodel, mlflow_run, X_test, y_test


def eval_model(**context):
    print("Eval Model")

    # load model and test data
    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    logmodel = task_instance_data[0]
    active_run = task_instance_data[1]
    X_test = task_instance_data[2]
    y_test = task_instance_data[3]

    mlflow.start_run(active_run.info.run_id)

    # check prediction report
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test, predictions))

    return logmodel, active_run


def register_model(**context):
    print("Register Model")
    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    logmodel = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    mlflow.sklearn.log_model(
        sk_model=logmodel,
        artifact_path='model',
        registered_model_name='intel_scenes_train_resnet50_lr'
    )

    mlflow.end_run()


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_train_resnet50_lr',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'training', 'logistic_regression', 'scikit_learn', 'pytorch', 'resnet50']
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        dag=dag,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
        provide_context=True
    )

    eval_model_task = PythonOperator(
        task_id='eval_model',
        python_callable=eval_model,
        dag=dag,
        provide_context=True
    )

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        dag=dag,
    )

    load_data_task >> train_model_task >> eval_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()
