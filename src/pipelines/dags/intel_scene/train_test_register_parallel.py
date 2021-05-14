import os
import torch
import pickle
import numpy as np
import pandas as pd

import sklearn.metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier

import matplotlib

import mlflow

from datetime import timedelta
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from fsdl_lib import feature_extraction as fe
from fsdl_lib.data import save_features
from fsdl_lib.data import load_features

from intel_scene.models_config import models_config

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

    model = kwargs['model']

    active_run = mlflow.start_run(experiment_id=experiment_id)
    mlflow.sklearn.autolog(log_models=False)

    model.fit(X_train, y_train)

    mlflow.log_param('train_features_hash', task_instance_data[1])
    mlflow.log_param('test_features_hash', task_instance_data[2])
    mlflow.log_param('features', kwargs['features'])
    mlflow.log_param('dag_type', 'parallel')

    y_pred = model.predict(X_test)
    mlflow.log_metric('test_accuracy_score',
                    sklearn.metrics.accuracy_score(y_test, y_pred))

    f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric('test_f1_score', f1_score)

    mlflow.log_metric('test_precision_score',
                    sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric('test_recall_score',
                    sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    mlflow.end_run()

    return active_run, model, f1_score


def register_best_model(**kwargs):
    print("Register Model")
    
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids=kwargs['task_ids'])

    best_f1_score = 0
    best_model = None
    best_run = None

    for mlflow_run , model , f1_score in task_instance_data:
        if f1_score > best_f1_score:
            best_model = model
            best_f1_score = f1_score
            best_run = mlflow_run
    
    mlflow.start_run(best_run.info.run_id)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path='model',
        registered_model_name='intel_scenes_train_resnet50'
    )

    mlflow.end_run()


def get_kwargs(model=None):
    '''
    generates kwargs for all model types
    '''
    return {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'experiment_name': 'intel_scene_images_parallel',
        'features': 'resnet50',
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

    register_best_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        op_kwargs={"task_ids": list(models_config.keys())},
        dag=dag
    )

    for task_id, model_args in models_config.items():

        model_class = model_args['model']
        model_hparams = model_args['hparams']
        
        train_test_task = PythonOperator(
            task_id=task_id,
            python_callable=train_test,
            op_kwargs=get_kwargs(model=model_class(**model_hparams)),
            dag=dag
        )

        create_dataset_task >> create_experiment_task >> train_test_task >> register_best_model_task

if __name__ == "__main__":
    dag.cli()
