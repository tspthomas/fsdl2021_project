import os
import torch
import pickle
import numpy as np
import pandas as pd

import sklearn.metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier

import matplotlib

import mlflow
from mlflow.tracking import MlflowClient

from datetime import timedelta
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from fsdl_lib import feature_extraction as fe
from fsdl_lib.data import save_features
from fsdl_lib.data import load_features

np.random.seed(33)

def create_dataset(**kwargs):
    print("Creating Dataset")

    print("Creating feature extractor")
    model_dir = '/opt/airflow/.cache/torch'
    model = fe.resnet50_feature_extractor(pretrained=True,
                                          model_dir=model_dir)
    transform = fe.get_transform()

    # paths
    src_path = os.path.join(kwargs['src'], kwargs['name'])
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])

    # train
    data_folder = 'seg_train_1000'
    train_features = fe.extract_features(src_path, dest_path, data_folder, model, transform)
    save_features(dest_path, data_folder, train_features)

    # test
    data_folder = 'seg_test_300'
    test_features =  fe.extract_features(src_path, dest_path, data_folder, model, transform)
    save_features(dest_path, data_folder, test_features)

def create_experiment(**kwargs):
    print("Create Experiment")

    t = datetime.now()
    experiment_id = mlflow.create_experiment(f"intel_scenes_compare_models_{t}")

    return experiment_id

def train_test(**kwargs):
    #train
    train_data_folder = 'seg_train_1000'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    train_features = load_features(dest_path, train_data_folder)

    X_train = train_features.X
    y_train = train_features.y

    #test
    test_data_folder = 'seg_test_300'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    test_features = load_features(dest_path, test_data_folder)

    X_test = test_features.X
    y_test = test_features.y

    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='create_experiment')
    experiment_id = task_instance_data[0]

    client = MlflowClient()

    model = kwargs['model']
    active_run = client.create_run(experiment_id)

    with mlflow.start_run(active_run.info.run_id):
        mlflow.sklearn.autolog(log_models=False)

        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        mlflow.log_metric('test_accuracy_score',
                        sklearn.metrics.accuracy_score(y_test, y_pred))

        f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric('test_f1_score', f1_score)

        mlflow.log_metric('test_precision_score',
                        sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('test_recall_score',
                        sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    print(f"Model: {model.get_params()}, f1_score: {f1_score}")
    return experiment_id, model, f1_score
    
def register_best_model(**kwargs):
    print("Register Model")
    
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids=kwargs['task_ids'])

    experiment_id = task_instance_data[0][0]
    best_f1_score = 0
    best_model = None

    for _ , model , f1_score in task_instance_data:
        if f1_score > best_f1_score:
            best_model = model
    
    client = MlflowClient()
    active_run = client.create_run(experiment_id)
    with mlflow.start_run(active_run.info.run_id):

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model',
            registered_model_name='intel_scenes_train_experiment'
        )

def get_kwargs(model=None):
    '''
    generates kwargs for all model types
    '''
    return {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'name': 'intel_scene_images',
        'model': model
    }

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_train_test_register_parallel',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'training', 'logistic_regression', 'scikit_learn', 'pytorch', 'resnet50', 'end_to_end']
) as dag:

    create_dataset_task = PythonOperator(
        task_id='create_dataset',
        python_callable=create_dataset,
        op_kwargs=get_kwargs(),
        dag=dag
    )

    create_experiment_task = PythonOperator(
        task_id='create_experiment',
        python_callable=create_experiment,
        op_kwargs=get_kwargs(),
        dag=dag
    )

    train_test_LogisticRegression_100_task = PythonOperator(
        task_id='train_test_LogisticRegression_100',
        python_callable=train_test,
        op_kwargs=get_kwargs(model=LogisticRegression(max_iter=100)),
        dag=dag
    )

    train_test_LogisticRegression_500_task = PythonOperator(
        task_id='train_test_LogisticRegression_500',
        python_callable=train_test,
        op_kwargs=get_kwargs(model=LogisticRegression(max_iter=500)),
        dag=dag
    )

    train_test_SGD_100_task = PythonOperator(
        task_id='train_test_SGD_100',
        python_callable=train_test,
        op_kwargs=get_kwargs(model=SGDClassifier(max_iter=100)),
        dag=dag
    )

    train_test_SGD_500_task = PythonOperator(
        task_id='train_test_SGD_500',
        python_callable=train_test,
        op_kwargs=get_kwargs(model=SGDClassifier(max_iter=500)),
        dag=dag
    )

    register_best_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        op_kwargs={"task_ids": [
            'train_test_LogisticRegression_100',
            'train_test_LogisticRegression_500',
            'train_test_SGD_100',
            'train_test_SGD_500'
        ]},
        dag=dag
    )

    create_dataset_task >> create_experiment_task >> train_test_LogisticRegression_100_task >> register_best_model_task
    create_dataset_task >> create_experiment_task >> train_test_LogisticRegression_500_task >> register_best_model_task
    create_dataset_task >> create_experiment_task >> train_test_SGD_100_task >> register_best_model_task
    create_dataset_task >> create_experiment_task >> train_test_SGD_500_task >> register_best_model_task


if __name__ == "__main__":
    dag.cli()
