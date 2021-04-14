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

from intel_scene.utils import resnet50_feature_extractor, create_np_arrays, RAW_DATA_DIR, PROCESSED_DATA_DIR

np.random.seed(33)

def create_dataset():
    print("Creating Dataset")

    # train
    _, X_train, y_train = create_np_arrays(dataset_name="train")

    with open(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train)

    with open(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train)

    print("Saved Train Data")


    # test
    _, X_test, y_test = create_np_arrays(dataset_name="test")
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test)

    with open(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test)

    print("Saved Test Data")

    return 

def train_model(**context):
    print("Train Model")

    with open(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), 'rb') as f:
        X_train = np.load(f)

    with open(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f)

    mlflow_run =  mlflow.start_run()    
    mlflow.sklearn.autolog(log_models=True)

    max_iter = 10000
    logmodel = LogisticRegression(max_iter=max_iter)
    logmodel.fit(X_train,y_train)

    return logmodel, mlflow_run


def eval_model(**context):
    print("Eval Model")

    with open(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), 'rb') as f:
        X_test = np.load(f)

    with open(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), 'rb') as f:
        y_test = np.load(f)

    # load model
    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    logmodel = task_instance_data[0]
    active_run = task_instance_data[1]

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

    # DEBUG, proving artifacts are written, however, not showing in mlflow UI
    client = mlflow.tracking.MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(active_run.info.run_id, "sklearn-logmodel")]
    print("artifacts {}".format(artifacts))
    print("mlflow.get_artifact_uri {}".format(mlflow.get_artifact_uri()))

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


    create_dataset_task = PythonOperator(
        task_id='load_data_and_preprocess', 
        python_callable=create_dataset, 
        dag=dag,
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

    create_dataset_task >> train_model_task >> eval_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()