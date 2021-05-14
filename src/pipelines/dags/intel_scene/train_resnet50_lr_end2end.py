import os
import torch
import pickle
import numpy as np
import pandas as pd

import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib

import mlflow

from datetime import timedelta

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


def train_model(**kwargs):
    print("Train Model")

    data_folder = 'seg_train_1000'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    train_features = load_features(dest_path, data_folder)

    X_train = train_features.X
    y_train = train_features.y

    # create or get existing experiment
    experiment = mlflow.get_experiment_by_name(kwargs['experiment_name'])

    if experiment:
        print("Experiment already existings. Using the same one.")
        experiment_id = experiment.experiment_id
    else:
        print("Experiment does not exist. Creating new one")
        experiment_id = mlflow.create_experiment(kwargs['experiment_name'])     

    mlflow_run = mlflow.start_run(experiment_id=experiment_id)
    mlflow.sklearn.autolog(log_models=False)

    logmodel = LogisticRegression(random_state=kwargs['random_state'])
    logmodel.fit(X_train, y_train)

    # log extra info
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='create_dataset')
    mlflow.log_param('train_features_hash', task_instance_data[0])
    mlflow.log_param('test_features_hash', task_instance_data[1])
    mlflow.log_param('features', kwargs['features'])
    mlflow.log_param('dag_type', 'default')

    return logmodel, mlflow_run


def eval_model(**kwargs):
    print("Eval Model")

    data_folder = 'seg_test_300'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    test_features = load_features(dest_path, data_folder)

    X_test = test_features.X
    y_test = test_features.y

    # load model
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    logmodel = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    # evaluate trained model
    y_pred = logmodel.predict(X_test)
    mlflow.log_metric('test_accuracy_score',
                      sklearn.metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric('test_f1_score',
                      sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric('test_precision_score',
                      sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric('test_recall_score',
                      sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    return logmodel, active_run


def register_model(**kwargs):
    print("Register Model")
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    logmodel = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    mlflow.sklearn.log_model(
        sk_model=logmodel,
        artifact_path='model',
        registered_model_name='intel_scenes_train_resnet50'
    )

    mlflow.end_run()


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_train_resnet50_lr_end2end',
    default_args=args,
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'training', 'logistic_regression', 'scikit_learn', 'pytorch', 'resnet50', 'end_to_end']
) as dag:

    dag_kwargs = {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'name': 'intel_scene_images',
        'random_state': 33,
        'experiment_name': 'intel_scene_images',
        'features': 'resnet50'
    }

    create_dataset_task = PythonOperator(
        task_id='create_dataset',
        python_callable=create_dataset,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    eval_model_task = PythonOperator(
        task_id='eval_model',
        python_callable=eval_model,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        dag=dag
    )

    create_dataset_task >> train_model_task >> eval_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()
