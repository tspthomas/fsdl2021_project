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

def train_model_LogisticRegression_100(**kwargs):
    print("Train Model")
    
    data_folder = 'seg_train_1000'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    train_features = load_features(dest_path, data_folder)

    X_train = train_features.X
    y_train = train_features.y

    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='create_experiment')
    experiment_id = task_instance_data[0]

    client = MlflowClient()

    model = LogisticRegression(max_iter=100)
    active_run = client.create_run(experiment_id)

    with mlflow.start_run(active_run.info.run_id):
        mlflow.sklearn.autolog(log_models=False)

        model.fit(X_train, y_train)

    return experiment_id, model

# def eval_model_LogisticRegression_100(**kwargs):
#     print("Eval Model")

#     data_folder = 'seg_test_300'
#     dest_path = os.path.join(kwargs['dest'], kwargs['name'])
#     test_features = load_features(dest_path, data_folder)

#     X_test = test_features.X
#     y_test = test_features.y

#     # load models and active runs
#     task_instance = kwargs['ti']
#     # task_instance_data = task_instance.xcom_pull(task_ids='train_models')
#     task_instance_data = task_instance.xcom_pull()
#     experiment_id = task_instance_data[0]
#     model = task_instance_data[1]

#     client = MlflowClient()

#     # evaluate trained models        
#     active_run = client.create_run(experiment_id)

#     with mlflow.start_run(active_run.info.run_id):

#         y_pred = model.predict(X_test)
#         mlflow.log_metric('test_accuracy_score',
#                         sklearn.metrics.accuracy_score(y_test, y_pred))

#         f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
#         mlflow.log_metric('test_f1_score', f1_score)

#         mlflow.log_metric('test_precision_score',
#                         sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
#         mlflow.log_metric('test_recall_score',
#                         sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

#     return experiment_id, model, f1_score

def train_model_LogisticRegression_500(**kwargs):
    print("Train Model")
    
    data_folder = 'seg_train_1000'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    train_features = load_features(dest_path, data_folder)

    X_train = train_features.X
    y_train = train_features.y

    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='create_experiment')
    experiment_id = task_instance_data[0]

    client = MlflowClient()

    model = LogisticRegression(max_iter=500)
    active_run = client.create_run(experiment_id)

    with mlflow.start_run(active_run.info.run_id):
        mlflow.sklearn.autolog(log_models=False)

        model.fit(X_train, y_train)

    return experiment_id, model

# TODO: check if need more specific arguments
def eval_model(**kwargs):
    print("Eval Model")

    data_folder = 'seg_test_300'
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])
    test_features = load_features(dest_path, data_folder)

    X_test = test_features.X
    y_test = test_features.y

    # load models and active runs
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull() # TODO: check if this pulls from previous task, as its same function used
    experiment_id = task_instance_data[0]
    model = task_instance_data[1]

    client = MlflowClient()

    # evaluate trained models        
    active_run = client.create_run(experiment_id)

    with mlflow.start_run(active_run.info.run_id):

        y_pred = model.predict(X_test)
        mlflow.log_metric('test_accuracy_score',
                        sklearn.metrics.accuracy_score(y_test, y_pred))

        f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric('test_f1_score', f1_score)

        mlflow.log_metric('test_precision_score',
                        sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('test_recall_score',
                        sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    return experiment_id, model, f1_score

# TODO: why isn't this working ?
def register_best_model(**kwargs):
    # return
    print("Register Model")
    
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids = [
        'eval_model_LogisticRegression_100'
        'eval_model_LogisticRegression_500'
        ]
    )

    print(task_instance_data)
    # best_f1_score = 0
    # best_model = None
    # for _ , model , f1_score in task_instance_data:
    #     if f1_score > best_f1_score:
    #         best_model = model
    # experiment_id = task_instance_data[0]
    # models = task_instance_data[1]
    # f1_score = task_instance_data[2]

    # for f1_score in f1_scores:
    #     print(f'models {f1_score}')

    # best_model, best_f1_score = None, 0

    # for model, f1_score in models, f1_scores:
    #     if f1_score > best_f1_score:
    #         best_model = model
    
    # client = MlflowClient()
    # active_run = client.create_run(experiment_id)
    # with mlflow.start_run(active_run.info.run_id):

    #     mlflow.sklearn.log_model(
    #         sk_model=best_model,
    #         artifact_path='model',
    #         registered_model_name='intel_scenes_train_experiment'
    #     )

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_train_parallel_end2end',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'training', 'logistic_regression', 'scikit_learn', 'pytorch', 'resnet50', 'end_to_end']
) as dag:

    dataset_kwargs = {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'name': 'intel_scene_images'
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

    train_model_LogisticRegression_100_task = PythonOperator(
        task_id='train_model_LogisticRegression_100',
        python_callable=train_model_LogisticRegression_100,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    eval_model_LogisticRegression_100_task = PythonOperator(
        task_id='eval_model_LogisticRegression_100',
        python_callable=eval_model,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    train_model_LogisticRegression_500_task = PythonOperator(
        task_id='train_model_LogisticRegression_500',
        python_callable=train_model_LogisticRegression_500,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    eval_model_LogisticRegression_500_task = PythonOperator(
        task_id='eval_model_LogisticRegression_500',
        python_callable=eval_model,
        op_kwargs=dataset_kwargs,
        dag=dag
    )

    register_best_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        dag=dag
    )

    # create_dataset_task >> create_experiment_task >> train_models_task >> eval_models_task >> register_best_model_task
    create_dataset_task >> create_experiment_task >> train_model_LogisticRegression_100_task >> eval_model_LogisticRegression_100_task >> register_best_model_task
    create_dataset_task >> create_experiment_task >> train_model_LogisticRegression_500_task >> eval_model_LogisticRegression_500_task >> register_best_model_task

if __name__ == "__main__":
    dag.cli()
