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

    train_features_hash = train_features.get_hash()
    test_features_hash = test_features.get_hash()

    return train_features_hash, test_features_hash

def create_experiment(**kwargs):
    print("Create Experiment")

    # create or get existing experiment
    experiment = mlflow.get_experiment_by_name(kwargs['experiment_name'])

    if experiment:
        print("Experiment already existings. Using the same one.")
        experiment_id = experiment.experiment_id
    else:
        print("Experiment does not exist. Creating new one")
        experiment_id = mlflow.create_experiment(kwargs['experiment_name'])  

    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull()
    train_features_hash = task_instance_data[0]
    test_features_hash = task_instance_data[1]

    return experiment_id, train_features_hash, test_features_hash

def train_models(**kwargs):
    print("Train Models")
    
    data_folder = 'seg_train_1000'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    train_features = load_features(dest_path, data_folder)

    X_train = train_features.X
    y_train = train_features.y

    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='create_experiment')
    experiment_id = task_instance_data[0]

    client = MlflowClient()
    models = kwargs['models']

    for model in models:
        
        active_run = client.create_run(experiment_id)

        with mlflow.start_run(active_run.info.run_id):
            mlflow.sklearn.autolog(log_models=False)

            model.fit(X_train, y_train)

            mlflow.log_param('train_features_hash', task_instance_data[1])
            mlflow.log_param('test_features_hash', task_instance_data[2])
            mlflow.log_param('features', kwargs['features'])

    return experiment_id, models


def test_models(**kwargs):
    print("Test Model")

    data_folder = 'seg_test_300'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    test_features = load_features(dest_path, data_folder)

    X_test = test_features.X
    y_test = test_features.y

    # load models and active runs
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_models')
    experiment_id = task_instance_data[0]
    models = task_instance_data[1]

    client = MlflowClient()

    # test  models
    max_f1_score = 0
    for model in models:
        
        active_run = client.create_run(experiment_id)

        with mlflow.start_run(active_run.info.run_id):

            y_pred = model.predict(X_test)
            mlflow.log_metric('test_accuracy_score',
                            sklearn.metrics.accuracy_score(y_test, y_pred))
            
            f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('test_f1_score', f1_score)

            # pick best model
            if f1_score > max_f1_score:
                best_model = model

            mlflow.log_metric('test_precision_score',
                            sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric('test_recall_score',
                            sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    return experiment_id, best_model


def register_best_model(**kwargs):
    print("Register Model")
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='test_models')
    experiment_id = task_instance_data[0]
    best_model = task_instance_data[1]

    client = MlflowClient()
    active_run = client.create_run(experiment_id)
    with mlflow.start_run(active_run.info.run_id):

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model',
            registered_model_name='intel_scenes_train_experiment'
        )

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_train_test_register_sequence',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'training', 'logistic_regression', 'scikit_learn', 'pytorch', 'resnet50', 'end_to_end']
) as dag:

    dataset_kwargs = {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'name': 'intel_scene_images',
        'experiment_name': 'intel_scene_images',
        'features': 'restnet50',
        'models': [
            LogisticRegression(max_iter = 100),
            LogisticRegression(max_iter = 500),
            SGDClassifier(max_iter= 100),
            SGDClassifier(max_iter= 500)
        ]
    }

    create_dataset_task = PythonOperator(
        task_id='create_dataset',
        python_callable=create_dataset,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    create_experiment_task = PythonOperator(
        task_id='create_experiment',
        python_callable=create_experiment,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    train_models_task = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    test_models_task = PythonOperator(
        task_id='test_models',
        python_callable=test_models,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    register_best_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        dag=dag
    )

    create_dataset_task >> create_experiment_task >> train_models_task >> test_models_task >> register_best_model_task


if __name__ == "__main__":
    dag.cli()
